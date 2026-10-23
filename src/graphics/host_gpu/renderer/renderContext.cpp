#include "graphics/host_gpu/renderer/renderContext.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "libs/errno.h"

#include <algorithm>
#include <cstring>

#include <chrono>
#include <cstdio>
#include <mutex>
#include <unordered_map>
#include <vector>
namespace Libs::Graphics {

extern thread_local int g_ps5plus_inval_source;

namespace {
struct FaultDiag {
	std::mutex m;
	std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
	uint64_t reads = 0, writes = 0;
	std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> pages; // page -> (reads, writes)
	std::unordered_map<uint64_t, uint64_t> pcs;
	void Note(bool write, uint64_t vaddr, uint64_t pc) {
		std::scoped_lock lock {m};
		(write ? writes : reads)++;
		auto& p = pages[vaddr >> 16];
		(write ? p.second : p.first)++;
		pcs[pc]++;
		const auto now = std::chrono::steady_clock::now();
		if (now - last < std::chrono::seconds(2)) return;
		last = now;
		std::fprintf(stderr, "FAULTS 2s: reads=%llu writes=%llu pages64k=%zu pcs=%zu\n", (unsigned long long)reads, (unsigned long long)writes, pages.size(), pcs.size());
		std::vector<std::pair<uint64_t, std::pair<uint64_t, uint64_t>>> top(pages.begin(), pages.end());
		std::sort(top.begin(), top.end(), [](auto& l, auto& r) { return l.second.first + l.second.second > r.second.first + r.second.second; });
		for (size_t i = 0; i < std::min<size_t>(6, top.size()); i++)
			std::fprintf(stderr, "FAULTS   page=0x%llx0000 reads=%llu writes=%llu\n", (unsigned long long)top[i].first, (unsigned long long)top[i].second.first, (unsigned long long)top[i].second.second);
		std::vector<std::pair<uint64_t, uint64_t>> tp(pcs.begin(), pcs.end());
		std::sort(tp.begin(), tp.end(), [](auto& l, auto& r) { return l.second > r.second; });
		for (size_t i = 0; i < std::min<size_t>(5, tp.size()); i++)
			std::fprintf(stderr, "FAULTS   pc=0x%llx count=%llu\n", (unsigned long long)tp[i].first, (unsigned long long)tp[i].second);
		std::fflush(stderr);
		reads = writes = 0; pages.clear(); pcs.clear();
	}
};
FaultDiag g_fault_diag;
} // namespace

RenderContext::RenderContext(GraphicContext& graphics)
    : m_graphics(graphics), m_render_executor(*this), m_command_scheduler(*this, graphics),
      m_descriptor_heap(graphics, m_command_scheduler.GetMasterSemaphore()),
      m_pipeline_cache(graphics, &m_image_heap), m_sampler_cache(graphics),
      m_buffer_cache(graphics, m_command_scheduler, m_page_manager, m_texture_cache),
      m_texture_cache(graphics, m_command_scheduler, m_page_manager, m_buffer_cache),
      m_image_heap(*this, m_page_manager),
      m_gpu_timestamps(graphics, m_command_scheduler, *this) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
}

RenderContext::~RenderContext() {
	ShutdownGpu();
	m_command_scheduler.Shutdown();
}

void RenderContext::InitializeGpu(VideoOut::VideoOutDriver* video_out) {
	EXIT_IF(m_gpu != nullptr);
	m_video_out = video_out;
	std::lock_guard lock(m_gpu_mutex);
	m_gpu = std::make_unique<GuestGpu>(*this);
}

void RenderContext::ShutdownGpu() {
	if (m_gpu != nullptr) {
		m_gpu->Shutdown();
		std::unique_ptr<GuestGpu> gpu;
		{
			std::lock_guard lock(m_gpu_mutex);
			gpu.swap(m_gpu);
		}
		gpu.reset();
	}
	if (m_video_out != nullptr) {
		if (m_command_scheduler.Active()) {
			m_command_scheduler.Finish();
		}
		m_command_scheduler.DrainPriorityOperations();
		m_video_out = nullptr;
	}
}

GuestGpu& RenderContext::GetGpu() const {
	EXIT_IF(m_gpu == nullptr);
	return *m_gpu;
}

bool RenderContext::PostGpuCommand(Common::UniqueFunction<void>&& command) {
	std::lock_guard lock(m_gpu_mutex);
	return m_gpu != nullptr && m_gpu->TrySendCommand(std::move(command));
}

void RenderContext::NotifyGuestWrite() {
	std::lock_guard lock(m_gpu_mutex);
	if (m_gpu != nullptr) {
		m_gpu->NotifyGuestWrite();
	}
}

VideoOut::VideoOutDriver& RenderContext::GetVideoOut() const {
	EXIT_IF(m_video_out == nullptr);
	return *m_video_out;
}

bool RenderContext::HandleFault(PageFaultAccess access, uint64_t fault_vaddr, uint64_t fault_pc,
                                uint64_t* resume_pc) noexcept {
	// The host reports the faulting byte, not the instruction's access width. Both caches
	// resolve its page; guessing a width can cross the end of a valid guest mapping.
	constexpr uint64_t fault_size = 1;
	if (!IsMapped(fault_vaddr, fault_size) ||
	    !m_page_manager.IsTrackingFault(fault_vaddr, access)) {
		return false;
	}
	g_fault_diag.Note(access == PageFaultAccess::Write, fault_vaddr, fault_pc);
	if (access == PageFaultAccess::Write) {
		m_buffer_cache.InvalidateMemory(fault_vaddr, fault_size);
		g_ps5plus_inval_source = 1;
		m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
		g_ps5plus_inval_source = 0;
		m_image_heap.InvalidateMemory(fault_vaddr, fault_size);
	} else if (resume_pc == nullptr ||
	           !ShaderRecompiler::IR::ResumeSrtLoadFault(fault_pc, *resume_pc)) {
		m_buffer_cache.ReadMemory(fault_vaddr, fault_size);
	}
	return true;
}

bool RenderContext::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	m_buffer_cache.InvalidateMemory(vaddr, size);
	g_ps5plus_inval_source = 2;
	m_texture_cache.InvalidateMemory(vaddr, size);
	g_ps5plus_inval_source = 0;
	m_image_heap.InvalidateMemory(vaddr, size);
	return true;
}

bool RenderContext::ResolvesWriteFaults(uint64_t vaddr, uint64_t size) const {
	return IsMapped(vaddr, size) && m_page_manager.IsWatched(vaddr, size);
}

bool RenderContext::StoreAtCompletion(uint64_t vaddr, const void* data, uint64_t size) {
	if (!IsMapped(vaddr, size)) {
		std::memcpy(reinterpret_cast<void*>(vaddr), data, size);
		return true;
	}
	// Cached copies cannot be invalidated here. Unwatched pages have none that a write can
	// stale, and the store excludes a cache registering a watcher meanwhile.
	return m_page_manager.StoreUnwatched(vaddr, size, data);
}

bool RenderContext::IsMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges.Contains(vaddr, size);
}

void RenderContext::MapMemory(uint64_t vaddr, uint64_t size) {
	std::lock_guard lock(m_mapped_ranges_mutex);
	m_mapped_ranges.Add(vaddr, size);
	m_buffer_cache.QueueBdaSync(vaddr, size);
}

void RenderContext::UnmapMemory(uint64_t vaddr, uint64_t size) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	const auto unmap = [this, vaddr, size] {
		// Check cache ownership on the GPU thread. Guest-memory callbacks can still
		// access a range with no cached data, so they must finish before it is unmapped.
		if (m_command_scheduler.Active() &&
		    (m_buffer_cache.IsRegionRegistered(vaddr, size) ||
		     m_texture_cache.IsRegionRegistered(vaddr, size) ||
		     m_command_scheduler.HasPendingPriorityOperations())) {
			const auto tick = m_command_scheduler.CurrentTick();
			m_command_scheduler.Finish();
			m_command_scheduler.WaitPriorityOperations(tick);
		}
		// Timestamps left to this thread would otherwise land after the range is unmapped.
		m_gpu_timestamps.StoreRetries();
		m_buffer_cache.InvalidateMemory(vaddr, size);
		m_texture_cache.UnmapMemory(vaddr, size);
		m_image_heap.UnmapMemory(vaddr, size);
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Subtract(vaddr, size);
	};
	// Shutdown still owns the GPU while queued rendering drains, but its command lane no
	// longer accepts external work. Use the guest GPU's state for the teardown route.
	if (m_gpu == nullptr || m_gpu->IsStopping()) {
		unmap();
		return;
	}
	m_gpu->SendCommandSync(unmap);
}

void RenderContext::PrepareBda() {
	if (!m_bda_logged) {
		Log::WriteToConsoleAndLog("GPU: using buffer device address (BDA) shader memory access.\n");
		m_bda_logged = true;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	m_buffer_cache.SynchronizeBda(m_mapped_ranges);
	m_fault_process_pending = true;
}

// A shader reads a guest page without a cache buffer as zero and drops its stores there: the BDA
// fault only creates the buffer after the shader ran. Ranges proven before the dispatch get one.
void RenderContext::MakeAddressRangeResident(uint64_t vaddr, uint64_t size) {
	// Larger proofs stay on the fault route, as FindBuffers caps unbounded descriptor ranges.
	constexpr uint64_t MaxResidentRange = uint64_t {256} * 1024 * 1024;
	if (size > MaxResidentRange || !GuestRange {vaddr, size}.Valid()) {
		return;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	m_mapped_ranges.ForEachInRange(vaddr, size, [this](uint64_t start, uint64_t end) {
		(void)m_buffer_cache.FindBuffer(start, end - start);
	});
}

void RenderContext::RunGarbageCollector() {
	if (m_fault_process_pending) {
		m_fault_process_pending = false;
		m_buffer_cache.ProcessFaultBuffer();
	}
	m_texture_cache.ProcessDownloadImages();
	m_texture_cache.RunGarbageCollector();
	m_buffer_cache.RunGarbageCollector();
}

void RenderContext::AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it != m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.push_back({eq, event_id});
}

void RenderContext::DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it == m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.erase(it);
}

void RenderContext::TriggerInterrupt(int event_id, uint32_t context_id) {
	std::vector<InterruptEqRegistration> registrations;
	{
		Common::LockGuard lock(m_interrupt_mutex);
		for (const auto& registration: m_interrupt_eqs) {
			if (registration.event_id == event_id) {
				registrations.push_back(registration);
			}
		}
	}

	for (const auto& registration: registrations) {
		const auto result = LibKernel::EventQueue::KernelTriggerEvent(
		    registration.eq, static_cast<uintptr_t>(registration.event_id),
		    LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS,
		    reinterpret_cast<void*>(static_cast<uintptr_t>(context_id)));
		if (result == LibKernel::KERNEL_ERROR_EBADF || result == LibKernel::KERNEL_ERROR_ENOENT) {
			DeleteInterruptEq(registration.eq, registration.event_id);
			continue;
		}
		EXIT_NOT_IMPLEMENTED(result != OK);
	}
}

} // namespace Libs::Graphics
