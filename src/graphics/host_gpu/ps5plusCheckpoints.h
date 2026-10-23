#pragma once
// PS5Plus debug aid: NVIDIA diagnostic checkpoints (VK_NV_device_diagnostic_checkpoints) at guest
// pass markers, dispatches and draws, printed when the device is lost. Enabled by
// PS5PLUS_GPU_CHECKPOINTS.
#include <vulkan/vulkan.h>

#include <array>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace Libs::Graphics::Ps5PlusCheckpoints {

inline PFN_vkCmdSetCheckpointNV        g_cmd_set  = nullptr;
inline PFN_vkGetQueueCheckpointDataNV  g_get_data = nullptr;
inline VkQueue                         g_queue    = VK_NULL_HANDLE;
inline std::array<std::array<char, 192>, 32768> g_labels {};
inline uint32_t                        g_next     = 0;
inline std::array<char, 96>            g_pass {"(start)"};
inline std::mutex                      g_mutex;
inline std::atomic<bool>               g_dumped {false};
// First indirect draw record of labeled draws, copied by the GPU right before the draw.
inline constexpr uint32_t              CaptureSlots = 4096;
inline constexpr uint32_t              CaptureBytes = 32;
inline const uint8_t*                  g_capture    = nullptr;
inline std::array<uint32_t, CaptureSlots> g_capture_label {};

// PS5PLUS_SKIP_PS=hash[,hash...]: draws whose pixel shader hash is listed are not executed.
inline bool SkipPixelShader(uint64_t hash) {
	static const std::vector<uint64_t> skip = [] {
		std::vector<uint64_t> list;
		if (const char* env = std::getenv("PS5PLUS_SKIP_PS"); env != nullptr) {
			for (const char* p = env; *p != '\0';) {
				char*      end  = nullptr;
				const auto hash = std::strtoull(p, &end, 16);
				if (end == p) {
					break;
				}
				list.push_back(hash);
				p = *end == ',' ? end + 1 : end;
			}
		}
		return list;
	}();
	for (const auto h: skip) {
		if (h == hash) {
			return true;
		}
	}
	return false;
}

inline bool EnvFlag(const char* name) {
	return std::getenv(name) != nullptr;
}

inline bool Requested() {
	return std::getenv("PS5PLUS_GPU_CHECKPOINTS") != nullptr;
}

inline void Mark(VkCommandBuffer cmd, const char* format, ...) {
	if (g_cmd_set == nullptr || cmd == VK_NULL_HANDLE) {
		return;
	}
	std::scoped_lock lock {g_mutex};
	auto&            slot = g_labels[g_next++ % g_labels.size()];
	const int        used = std::snprintf(slot.data(), slot.size(), "#%u %s | ", g_next, g_pass.data());
	va_list          args;
	va_start(args, format);
	std::vsnprintf(slot.data() + used, slot.size() - static_cast<size_t>(used), format, args);
	va_end(args);
	g_cmd_set(cmd, slot.data());
}

inline void SetPass(VkCommandBuffer cmd, const char* name) {
	{
		std::scoped_lock lock {g_mutex};
		std::snprintf(g_pass.data(), g_pass.size(), "%s", name);
	}
	Mark(cmd, "marker");
}

inline void Dump(const char* where) {
	if (g_get_data == nullptr || g_queue == VK_NULL_HANDLE || g_dumped.exchange(true)) {
		return;
	}
	uint32_t count = 0;
	g_get_data(g_queue, &count, nullptr);
	std::vector<VkCheckpointDataNV> data(count);
	for (auto& d: data) {
		d.sType = VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV;
		d.pNext = nullptr;
	}
	g_get_data(g_queue, &count, data.data());
	std::fprintf(stderr, "PS5PLUS GPU CHECKPOINTS at %s: %u (last label #%u)\n", where, count, g_next);
	for (const auto& d: data) {
		std::fprintf(stderr, "PS5PLUS GPU CHECKPOINT stage=0x%08x %s\n", static_cast<unsigned>(d.stage),
		             d.pCheckpointMarker != nullptr ? static_cast<const char*>(d.pCheckpointMarker) : "?");
	}
	// Everything recorded after the last completed checkpoint up to the last started one.
	uint32_t low = UINT32_MAX, high = 0;
	for (const auto& d: data) {
		unsigned n = 0;
		if (d.pCheckpointMarker != nullptr &&
		    std::sscanf(static_cast<const char*>(d.pCheckpointMarker), "#%u", &n) == 1) {
			low  = n < low ? n : low;
			high = n > high ? n : high;
		}
	}
	const auto capture = [](uint32_t n) {
		const auto slot = n % CaptureSlots;
		if (g_capture == nullptr || g_capture_label[slot] != n) {
			return;
		}
		uint32_t words[5];
		std::memcpy(words, g_capture + static_cast<size_t>(slot) * CaptureBytes, sizeof(words));
		std::fprintf(stderr, "PS5PLUS GPU   args #%u: %08x %08x %08x %08x %08x\n", n, words[0], words[1],
		             words[2], words[3], words[4]);
	};
	for (uint32_t n = low > 32u ? low - 32u : 1u; low != UINT32_MAX && n <= low; n++) {
		std::fprintf(stderr, "PS5PLUS GPU BEFORE %s\n", g_labels[(n - 1u) % g_labels.size()].data());
		capture(n);
	}
	if (low != UINT32_MAX && high > low && high - low <= 256u) {
		for (uint32_t n = low + 1; n <= high; n++) {
			std::fprintf(stderr, "PS5PLUS GPU PENDING %s\n", g_labels[(n - 1u) % g_labels.size()].data());
			capture(n);
		}
	}
	// What was recorded right after the last started checkpoint.
	for (uint32_t n = high + 1; high != 0 && n <= g_next && n <= high + 48u; n++) {
		std::fprintf(stderr, "PS5PLUS GPU NEXT %s\n", g_labels[(n - 1u) % g_labels.size()].data());
	}
	std::fflush(stderr);
}

} // namespace Libs::Graphics::Ps5PlusCheckpoints
