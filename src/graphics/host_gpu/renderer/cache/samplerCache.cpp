#include "graphics/host_gpu/renderer/cache/samplerCache.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace Libs::Graphics {
namespace {

// Samplers created outside the cache, such as the overlay's, also count against the device limit.
constexpr size_t SamplerLimitReserve = 64;

} // namespace

SamplerCache::~SamplerCache() {
	for (const auto& [key, entry]: m_samplers) {
		(void)key;
		m_graphics.device.destroySampler(entry.sampler, nullptr);
	}
}

// The device limits live samplers (maxSamplerAllocationCount is 4000 on NVIDIA), but a game can
// name far more distinct sampler descriptors over a session, for example by moving LOD bias with
// its dynamic resolution. Destroy the least recently used quarter once the GPU is done with it.
void SamplerCache::Reclaim() {
	const uint64_t current = m_master_semaphore.CurrentTick();

	std::vector<std::pair<uint64_t, SamplerKey>> victims;
	victims.reserve(m_samplers.size());
	for (const auto& [key, entry]: m_samplers) {
		// Samplers used by the command buffer being recorded cannot be waited for yet.
		if (entry.last_tick < current) {
			victims.emplace_back(entry.last_tick, key);
		}
	}
	const size_t count = std::min(victims.size(), std::max<size_t>(m_samplers.size() / 4, 1));
	if (count == 0) {
		return;
	}
	std::ranges::nth_element(victims, victims.begin() + static_cast<std::ptrdiff_t>(count - 1));
	victims.resize(count);

	uint64_t newest = 0;
	for (const auto& [tick, key]: victims) {
		newest = std::max(newest, tick);
	}
	m_master_semaphore.Wait(newest);

	for (const auto& [tick, key]: victims) {
		(void)tick;
		auto iter = m_samplers.find(key);
		m_graphics.device.destroySampler(iter->second.sampler, nullptr);
		m_samplers.erase(iter);
	}
	LOGF("SamplerCache: destroyed %zu least recently used samplers, %zu remain\n", count,
	     m_samplers.size());
}

vk::Sampler SamplerCache::GetSampler(const ShaderSamplerResource& r, bool integer_border) {
	Common::LockGuard lock(m_mutex);

	const SamplerKey key {r.fields[0], r.fields[1], r.fields[2], r.fields[3], integer_border};
	const uint64_t   tick = m_master_semaphore.CurrentTick();
	if (auto iter = m_samplers.find(key); iter != m_samplers.end()) {
		iter->second.last_tick = tick;
		return iter->second.sampler;
	}

	if (m_capacity == 0) {
		const size_t limit =
		    m_graphics.GetPhysicalDeviceProperties().limits.maxSamplerAllocationCount;
		m_capacity = std::max(limit, 2 * SamplerLimitReserve) - SamplerLimitReserve;
		m_capacity = std::min<size_t>(m_capacity, 300); // DEBUG ONLY: exercise Reclaim()
	}
	if (m_samplers.size() >= m_capacity) {
		Reclaim();
	}

	float      aniso_ratio = 1.0f;
	const auto mag_filter  = r.XyMagFilter();
	const auto min_filter  = r.XyMinFilter();

	auto is_aniso_filter = [](uint8_t filter) {
		switch (static_cast<Prospero::SamplerFilter>(filter)) {
			case Prospero::SamplerFilter::kAnisoPoint:
			case Prospero::SamplerFilter::kAnisoLinear: return true;
			case Prospero::SamplerFilter::kPoint:
			case Prospero::SamplerFilter::kBilinear: return false;
			default: EXIT("unknown sampler filter: %u\n", filter);
		}
		return false;
	};

	auto to_vk_filter = [](uint8_t filter) {
		switch (static_cast<Prospero::SamplerFilter>(filter)) {
			case Prospero::SamplerFilter::kPoint:
			case Prospero::SamplerFilter::kAnisoPoint: return vk::Filter::eNearest;
			case Prospero::SamplerFilter::kBilinear:
			case Prospero::SamplerFilter::kAnisoLinear: return vk::Filter::eLinear;
			default: EXIT("unknown sampler filter: %u\n", filter);
		}
		return vk::Filter::eNearest;
	};

	const bool aniso = is_aniso_filter(mag_filter) || is_aniso_filter(min_filter);
	if (aniso) {
		switch (static_cast<Prospero::SamplerAnisoRatio>(r.MaxAnisoRatio())) {
			case Prospero::SamplerAnisoRatio::kOne: aniso_ratio = 1.0f; break;
			case Prospero::SamplerAnisoRatio::kTwo: aniso_ratio = 2.0f; break;
			case Prospero::SamplerAnisoRatio::kFour: aniso_ratio = 4.0f; break;
			case Prospero::SamplerAnisoRatio::kEight: aniso_ratio = 8.0f; break;
			case Prospero::SamplerAnisoRatio::kSixteen:
			// Reserved encodings 5-7 fall back to the maximum ratio.
			default: aniso_ratio = 16.0f; break;
		}
	}

	const auto mip_filter = r.MipFilter();
	float      min_lod    = 0.0f;
	float      max_lod    = 0.0f;
	if (static_cast<Prospero::SamplerMipFilter>(mip_filter) != Prospero::SamplerMipFilter::kNone) {
		min_lod = static_cast<float>(r.MinLod()) / 256.0f;
		max_lod = static_cast<float>(r.MaxLod()) / 256.0f;
	}

	vk::SamplerCreateInfo sampler_info {};

	auto to_vk_address_mode = [](uint8_t clamp) {
		switch (static_cast<Prospero::SamplerClampMode>(clamp)) {
			case Prospero::SamplerClampMode::kWrap: return vk::SamplerAddressMode::eRepeat;
			case Prospero::SamplerClampMode::kMirror:
				return vk::SamplerAddressMode::eMirroredRepeat;
			case Prospero::SamplerClampMode::kClampLastTexel:
				return vk::SamplerAddressMode::eClampToEdge;
			case Prospero::SamplerClampMode::kMirrorOnceLastTexel:
				return vk::SamplerAddressMode::eMirrorClampToEdge;
			case Prospero::SamplerClampMode::kClampHalfBorder:
				return vk::SamplerAddressMode::eClampToBorder;
			case Prospero::SamplerClampMode::kMirrorOnceHalfBorder:
				return vk::SamplerAddressMode::eMirrorClampToEdge;
			case Prospero::SamplerClampMode::kClampBorder:
				return vk::SamplerAddressMode::eClampToBorder;
			case Prospero::SamplerClampMode::kMirrorOnceBorder:
				return vk::SamplerAddressMode::eMirrorClampToEdge;
			default: EXIT("unknown clamp: %u\n", clamp);
		}
		return vk::SamplerAddressMode::eClampToBorder;
	};

	vk::BorderColor border = integer_border ? vk::BorderColor::eIntTransparentBlack
	                                       : vk::BorderColor::eFloatTransparentBlack;
	switch (static_cast<Prospero::SamplerBorderColor>(r.BorderColorType())) {
		case Prospero::SamplerBorderColor::kTransBlack: break;
		case Prospero::SamplerBorderColor::kOpaqueBlack:
			border = integer_border ? vk::BorderColor::eIntOpaqueBlack
			                        : vk::BorderColor::eFloatOpaqueBlack;
			break;
		case Prospero::SamplerBorderColor::kOpaqueWhite:
			border = integer_border ? vk::BorderColor::eIntOpaqueWhite
			                        : vk::BorderColor::eFloatOpaqueWhite;
			break;
		case Prospero::SamplerBorderColor::kFromTable:
			LOGF(
			    "temporary: approximating table border color as transparent black, index = %" PRIu16
			    "\n",
			    r.BorderColorPtr());
			break;
		default: EXIT("unknown border color: %d", static_cast<int>(r.BorderColorType()));
	}

	sampler_info.magFilter = to_vk_filter(mag_filter);
	sampler_info.minFilter = to_vk_filter(min_filter);
	sampler_info.mipmapMode =
	    (static_cast<Prospero::SamplerMipFilter>(mip_filter) == Prospero::SamplerMipFilter::kLinear
	         ? vk::SamplerMipmapMode::eLinear
	         : vk::SamplerMipmapMode::eNearest);
	sampler_info.addressModeU = to_vk_address_mode(r.ClampX());
	sampler_info.addressModeV = to_vk_address_mode(r.ClampY());
	sampler_info.addressModeW = to_vk_address_mode(r.ClampZ());
    const auto max_lod_bias = m_graphics.GetPhysicalDeviceProperties().limits.maxSamplerLodBias;
    sampler_info.mipLodBias = std::clamp(
        static_cast<float>(static_cast<int16_t>((r.LodBias() ^ 0x2000u) - 0x2000u)) / 256.0f,
        -max_lod_bias, max_lod_bias
    );
	sampler_info.anisotropyEnable        = (aniso ? VK_TRUE : VK_FALSE);
	sampler_info.maxAnisotropy           = aniso_ratio;
	sampler_info.compareEnable           = (r.DepthCompareFunc() != 0 ? VK_TRUE : VK_FALSE);
	sampler_info.compareOp               = static_cast<vk::CompareOp>(r.DepthCompareFunc());
	sampler_info.minLod                  = min_lod;
	sampler_info.maxLod                  = max_lod;
	sampler_info.borderColor             = border;
	sampler_info.unnormalizedCoordinates = (r.ForceUnormCoords() ? VK_TRUE : VK_FALSE);

	if (r.ForceUnormCoords()) {
		sampler_info.addressModeU     = vk::SamplerAddressMode::eClampToEdge;
		sampler_info.addressModeV     = vk::SamplerAddressMode::eClampToEdge;
		sampler_info.addressModeW     = vk::SamplerAddressMode::eClampToEdge;
		sampler_info.mipmapMode       = vk::SamplerMipmapMode::eNearest;
		sampler_info.minLod           = 0.0f;
		sampler_info.maxLod           = 0.0f;
		sampler_info.anisotropyEnable = VK_FALSE;
		sampler_info.maxAnisotropy    = 1.0f;
		sampler_info.compareEnable    = VK_FALSE;
		sampler_info.mipLodBias       = 0.0f;
	}

	vk::Sampler vk_sampler = nullptr;
	const auto  result     = m_graphics.device.createSampler(&sampler_info, nullptr, &vk_sampler);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || vk_sampler == nullptr);

	m_samplers.emplace(key, Entry {vk_sampler, tick});
	return vk_sampler;
}

} // namespace Libs::Graphics
