#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SAMPLERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SAMPLERCACHE_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/shaderBindings.h"

#include <array>
#include <cstddef>
#include <unordered_map>

namespace Libs::Graphics {

struct GraphicContext;
class MasterSemaphore;

class SamplerCache {
public:
	SamplerCache(GraphicContext& graphics, MasterSemaphore& master_semaphore)
	    : m_graphics(graphics), m_master_semaphore(master_semaphore) {
		EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	}
	~SamplerCache();
	KYTY_CLASS_NO_COPY(SamplerCache);

	vk::Sampler GetSampler(const ShaderSamplerResource& r, bool integer_border);

private:
	using SamplerKey = std::array<uint32_t, 5>;

	struct Entry {
		vk::Sampler sampler   = nullptr;
		uint64_t    last_tick = 0;
	};

	void Reclaim();

	struct SamplerKeyHash {
		std::size_t operator()(const SamplerKey& key) const {
			std::size_t hash = 0;
			for (auto value: key) {
				hash ^= static_cast<std::size_t>(value) +
				        static_cast<std::size_t>(0x9e3779b97f4a7c15ull) + (hash << 6u) +
				        (hash >> 2u);
			}
			return hash;
		}
	};

	GraphicContext&                                       m_graphics;
	MasterSemaphore&                                      m_master_semaphore;
	Common::Mutex                                         m_mutex;
	std::unordered_map<SamplerKey, Entry, SamplerKeyHash> m_samplers;
	size_t                                                m_capacity = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SAMPLERCACHE_H_
