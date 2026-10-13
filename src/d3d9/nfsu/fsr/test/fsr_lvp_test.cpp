// NFSU FSR: native Vulkan test of the generated FSR 3.1 passes (nfsu_fsr_spv.h) on any desktop driver (lavapipe here),
// with the Khronos validation layer. It mirrors d3d9_nfsu_fsr.cpp (same images, formats, bindings, constants, clears,
// pass order) and renders a synthetic HDR scene: a rotated checker + thin lines, panning right v px/frame, point-sampled
// at jittered pixel centres (= aliased input). After N frames it compares FSR's output and the raw input against an
// unjittered 8x8 supersampled reference (RMSE).
// Usage: fsr_lvp_test [frames] [jitterSign +1/-1] [sharpness, <0 = no RCAS] [nowave 0/1] [MV multiplier] [out.ppm|-] [box: -1/0/1]
//                     [upscale ratio display/render, 1 = Native AA] [fp16 0/1: the _h modules, needs shaderFloat16]
// Upscaling: the scene lives in DISPLAY pixels; render pixel (x, y) samples it at ((x + 0.5 - jx) * ratio, ...); the "input"
// RMSE is a bilinear upscale of the render image (what plain stretching gives).
// Build: g++ -O2 -std=c++17 fsr_lvp_test.cpp -I.. -lvulkan -o fsr_lvp_test
#include <vulkan/vulkan.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../nfsu_fsr_spv.h"

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { std::printf("FAIL %s = %d (line %d)\n", #x, r_, __LINE__); std::exit(1); } } while (0)

static int g_valErrors = 0;
static VKAPI_ATTR VkBool32 VKAPI_CALL dbg(VkDebugUtilsMessageSeverityFlagBitsEXT sev, VkDebugUtilsMessageTypeFlagsEXT,
                                          const VkDebugUtilsMessengerCallbackDataEXT* d, void*) {
  if (sev >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    if (sev >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) g_valErrors++;
    if (g_valErrors < 20) std::printf("[validation %s] %s\n", sev >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT ? "ERROR" : "warn", d->pMessage);
  }
  return VK_FALSE;
}

static VkDevice dev; static VkPhysicalDevice phys; static VkQueue queue; static uint32_t qfam;
static VkPhysicalDeviceMemoryProperties memProps;

static uint32_t memType(uint32_t bits, VkMemoryPropertyFlags f) {
  for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
    if ((bits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & f) == f) return i;
  std::printf("no memory type\n"); std::exit(1);
}

struct Img { VkImage img; VkDeviceMemory mem; VkImageView sv, rv; VkImageView mip[6]; VkFormat fmt; uint32_t w, h, mips; };
static Img mkImg(VkFormat fmt, uint32_t w, uint32_t h, uint32_t mips = 1, bool storage = true) {
  Img r = {}; r.fmt = fmt; r.w = w; r.h = h; r.mips = mips;
  VkImageCreateInfo ci = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
  ci.imageType = VK_IMAGE_TYPE_2D; ci.format = fmt; ci.extent = { w, h, 1 }; ci.mipLevels = mips; ci.arrayLayers = 1;
  ci.samples = VK_SAMPLE_COUNT_1_BIT; ci.tiling = VK_IMAGE_TILING_OPTIMAL;
  ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | (storage ? VK_IMAGE_USAGE_STORAGE_BIT : 0);
  CK(vkCreateImage(dev, &ci, nullptr, &r.img));
  VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev, r.img, &mr);
  VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO }; ai.allocationSize = mr.size;
  ai.memoryTypeIndex = memType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  CK(vkAllocateMemory(dev, &ai, nullptr, &r.mem)); CK(vkBindImageMemory(dev, r.img, r.mem, 0));
  VkImageViewCreateInfo vi = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
  vi.image = r.img; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = fmt;
  vi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, 1 };
  CK(vkCreateImageView(dev, &vi, nullptr, &r.rv));
  vi.subresourceRange.levelCount = 1;
  if (storage) CK(vkCreateImageView(dev, &vi, nullptr, &r.sv));
  if (mips > 1) for (uint32_t m = 0; m < 6; m++) { vi.subresourceRange.baseMipLevel = m < mips ? m : mips - 1; CK(vkCreateImageView(dev, &vi, nullptr, &r.mip[m])); }
  return r;
}

struct Buf { VkBuffer buf; VkDeviceMemory mem; void* map; VkDeviceSize size; };
static Buf mkBuf(VkDeviceSize size, VkBufferUsageFlags usage) {
  Buf b = {}; b.size = size;
  VkBufferCreateInfo ci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO }; ci.size = size; ci.usage = usage;
  CK(vkCreateBuffer(dev, &ci, nullptr, &b.buf));
  VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, b.buf, &mr);
  VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO }; ai.allocationSize = mr.size;
  ai.memoryTypeIndex = memType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  CK(vkAllocateMemory(dev, &ai, nullptr, &b.mem)); CK(vkBindBufferMemory(dev, b.buf, b.mem, 0));
  CK(vkMapMemory(dev, b.mem, 0, size, 0, &b.map));
  return b;
}

struct Pipe { VkDescriptorSetLayout dsl; VkPipelineLayout pl; VkPipeline p; const VkDescriptorType* types; uint32_t n; };
static Pipe mkPipe(const uint32_t* code, size_t size, const VkDescriptorType* types, uint32_t n) {
  Pipe r = {}; r.types = types; r.n = n;
  std::vector<VkDescriptorSetLayoutBinding> b(n);
  for (uint32_t i = 0; i < n; i++) b[i] = { i, types[i], 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
  VkDescriptorSetLayoutCreateInfo dci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO }; dci.bindingCount = n; dci.pBindings = b.data();
  CK(vkCreateDescriptorSetLayout(dev, &dci, nullptr, &r.dsl));
  VkPipelineLayoutCreateInfo pci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO }; pci.setLayoutCount = 1; pci.pSetLayouts = &r.dsl;
  CK(vkCreatePipelineLayout(dev, &pci, nullptr, &r.pl));
  VkShaderModuleCreateInfo mci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO }; mci.codeSize = size; mci.pCode = code;
  VkShaderModule m; CK(vkCreateShaderModule(dev, &mci, nullptr, &m));
  VkComputePipelineCreateInfo cpi = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
  cpi.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, m, "main", nullptr };
  cpi.layout = r.pl;
  CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &r.p));
  return r;
}

// same layout as d3d9_nfsu_fsr.cpp
struct FsrConstants {
  int32_t renderSize[2], previousFrameRenderSize[2], upscaleSize[2], previousFrameUpscaleSize[2], maxRenderSize[2], maxUpscaleSize[2];
  float deviceToViewDepth[4], jitterOffset[2], previousFrameJitterOffset[2], motionVectorScale[2], downscaleFactor[2];
  float motionVectorJitterCancellation[2], tanHalfFOV, jitterPhaseCount, deltaTime, deltaPreExposure, viewSpaceToMetersFactor, frameIndex;
  float velocityFactor, reactivenessScale, shadingChangeScale, accumulationAddedPerFrame, minDisocclusionAccumulation, pad[3];
};
struct SpdC { uint32_t mips, numWorkGroups, workGroupOffset[2], renderSize[2], pad[2]; };

static float halton(unsigned i, unsigned b) { float f = 1, r = 0; for (; i; i /= b) { f /= b; r += f * (i % b); } return r; }
static void scene(float x, float y, float* c) {                    // HDR test scene in pixel units (y down)
  const float a = 0.35f, cs = std::cos(a), sn = std::sin(a);
  float u = (x * cs + y * sn) / 7.3f, v = (-x * sn + y * cs) / 7.3f;
  int chk = (int(std::floor(u)) + int(std::floor(v))) & 1;
  float line = std::fabs(std::fmod(x * 0.8f - y * 0.6f + 1000.0f, 23.0f) - 11.5f) < 0.6f ? 1.0f : 0.0f;
  c[0] = chk ? 0.9f : 0.05f; c[1] = chk ? 0.6f : 0.08f; c[2] = chk ? 0.2f : 0.3f;
  if (line) { c[0] = 4.0f; c[1] = 3.0f; c[2] = 1.0f; }            // bright HDR lines (neon-like)
  c[3] = 1.0f;
}

// screen-space shading: background scene panning right at vel px/frame + optional "car" box moving LEFT at ovel px/frame
// (fine vertical stripes, nearer depth). obj = 1 inside the box.
static int g_objMode = -1; static const float g_ovel = 2.9f;
static void shadePx(float sx, float sy, int fr, float vel, float* c, int* obj) {
  float ox = 180.0f - g_ovel * fr;
  *obj = g_objMode >= 0 && sx >= ox && sx < ox + 60.0f && sy >= 50.0f && sy < 100.0f;
  if (!*obj) { scene(sx - vel * fr, sy, c); return; }
  bool st = std::fmod(sx - ox + 1000.0f, 2.5f) < 1.25f;
  c[0] = st ? 2.5f : 0.1f; c[1] = st ? 0.2f : 0.05f; c[2] = st ? 0.2f : 0.6f; c[3] = 1.0f;
}

int main(int argc, char** argv) {
  const int frames = argc > 1 ? std::atoi(argv[1]) : 32;
  const float jsign = argc > 2 ? float(std::atof(argv[2])) : 1.0f;
  const float sharp = argc > 3 ? float(std::atof(argv[3])) : -1.0f;
  const bool nowave = argc > 4 && std::atoi(argv[4]) != 0;
  const float mvmul = argc > 5 ? float(std::atof(argv[5])) : 1.0f;   // 0 = no motion vectors (ghosting check)
  const char* ppm = argc > 6 && argv[6][0] != '-' ? argv[6] : nullptr; // last frame: input | FSR | reference (tonemapped)
  g_objMode = argc > 7 ? std::atoi(argv[7]) : -1;                     // moving box: -1 none, 1 own MVs, 0 camera MVs only
  const float ratio = argc > 8 ? float(std::atof(argv[8])) : 1.0f;
  const bool fp16 = argc > 9 && std::atoi(argv[9]) != 0;
  const uint32_t W = 256, H = 144;                                   // render size
  const uint32_t OW = uint32_t(W * ratio + 0.5f), OH = uint32_t(H * ratio + 0.5f);   // display size
  const int phases = int(8.0f * std::pow(float(OW) / W, 2.0f));     // ffxFsr3UpscalerGetJitterPhaseCount
  const float vel = 1.37f;                                          // px per frame, content moves right

  VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO }; app.apiVersion = VK_API_VERSION_1_2;
  const char* layers[] = { "VK_LAYER_KHRONOS_validation" }; const char* exts[] = { VK_EXT_DEBUG_UTILS_EXTENSION_NAME };
  VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO }; ici.pApplicationInfo = &app;
  ici.enabledLayerCount = 1; ici.ppEnabledLayerNames = layers; ici.enabledExtensionCount = 1; ici.ppEnabledExtensionNames = exts;
  VkInstance inst; CK(vkCreateInstance(&ici, nullptr, &inst));
  auto cdm = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(inst, "vkCreateDebugUtilsMessengerEXT");
  VkDebugUtilsMessengerCreateInfoEXT dmi = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
  dmi.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  dmi.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT; dmi.pfnUserCallback = dbg;
  VkDebugUtilsMessengerEXT msg; CK(cdm(inst, &dmi, nullptr, &msg));
  uint32_t n = 1; CK(vkEnumeratePhysicalDevices(inst, &n, &phys) == VK_INCOMPLETE ? VK_SUCCESS : VK_SUCCESS);
  vkGetPhysicalDeviceMemoryProperties(phys, &memProps);
  VkPhysicalDeviceSubgroupProperties sg = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES };
  VkPhysicalDeviceProperties2 pp = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 }; pp.pNext = &sg;
  vkGetPhysicalDeviceProperties2(phys, &pp);
  bool wave = !nowave && (sg.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) && (sg.supportedOperations & VK_SUBGROUP_FEATURE_QUAD_BIT);
  std::printf("device %s, subgroup size %u ops 0x%x -> SPD %s\n", pp.properties.deviceName, sg.subgroupSize, sg.supportedOperations, wave ? "quad ops" : "no wave ops");
  float prio = 1; qfam = 0;
  VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO }; qci.queueFamilyIndex = 0; qci.queueCount = 1; qci.pQueuePriorities = &prio;
  VkPhysicalDeviceVulkan12Features f12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
  VkPhysicalDeviceFeatures2 f2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 }; f2.pNext = &f12;
  vkGetPhysicalDeviceFeatures2(phys, &f2);
  std::printf("shaderFloat16 %u shaderInt16 %u\n", f12.shaderFloat16, f2.features.shaderInt16);
  if (fp16 && (!f12.shaderFloat16 || !f2.features.shaderInt16)) { std::printf("fp16 not supported here\n"); return 3; }
  VkPhysicalDeviceVulkan12Features e12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES }; e12.shaderFloat16 = fp16;
  VkPhysicalDeviceFeatures feat = {};                              // like DXVK 3.x: no storage-image format features enabled
  feat.shaderInt16 = fp16;
  VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO }; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci; dci.pEnabledFeatures = &feat;
  if (fp16) dci.pNext = &e12;
  CK(vkCreateDevice(phys, &dci, nullptr, &dev)); vkGetDeviceQueue(dev, 0, 0, &queue);

  using namespace dxvk::nfsu_fsr;
#define P(x) (fp16 ? mkPipe(x##_h_spv, sizeof(x##_h_spv), x##_h_types, sizeof(x##_h_types) / sizeof(x##_h_types[0])) \
                    : mkPipe(x##_spv, sizeof(x##_spv), x##_types, sizeof(x##_types) / sizeof(x##_types[0])))
  Pipe pPrep = P(prepare_inputs), pLuma = wave ? P(luma_pyramid) : P(luma_pyramid_nowave);
  Pipe pScp = wave ? P(shading_change_pyramid) : P(shading_change_pyramid_nowave), pSc = P(shading_change), pReact = P(prepare_reactivity);
  Pipe pInst = P(luma_instability), pAcc = P(accumulate), pAccS = P(accumulate_sharpen), pRcas = P(rcas);
  std::printf("12 modules -> 9 pipelines created\n");

  const uint32_t hw = W / 2, hh = H / 2; uint32_t spdMips = 1; for (uint32_t m = hw > hh ? hw : hh; m > 1; m >>= 1) spdMips++;
  Img dilMv = mkImg(VK_FORMAT_R16G16_SFLOAT, W, H), dilDepth = mkImg(VK_FORMAT_R32_SFLOAT, W, H), recDepth = mkImg(VK_FORMAT_R32_UINT, W, H);
  Img inter = mkImg(VK_FORMAT_R16_SFLOAT, W, H), luma[2] = { mkImg(VK_FORMAT_R16_SFLOAT, W, H), mkImg(VK_FORMAT_R16_SFLOAT, W, H) };
  Img atom = mkImg(VK_FORMAT_R32_UINT, 1, 1), finfo = mkImg(VK_FORMAT_R32G32B32A32_SFLOAT, 1, 1), spd = mkImg(VK_FORMAT_R16G16_SFLOAT, hw, hh, spdMips);
  Img far1 = mkImg(VK_FORMAT_R16_SFLOAT, hw, hh), shade = mkImg(VK_FORMAT_R8_UNORM, hw, hh), dreact = mkImg(VK_FORMAT_R8G8B8A8_UNORM, W, H);
  Img locks = mkImg(VK_FORMAT_R8_UNORM, OW, OH), acc[2] = { mkImg(VK_FORMAT_R8_UNORM, W, H), mkImg(VK_FORMAT_R8_UNORM, W, H) };
  Img lh[2] = { mkImg(VK_FORMAT_R16G16B16A16_SFLOAT, W, H), mkImg(VK_FORMAT_R16G16B16A16_SFLOAT, W, H) };
  Img up[2] = { mkImg(VK_FORMAT_R16G16B16A16_SFLOAT, OW, OH), mkImg(VK_FORMAT_R16G16B16A16_SFLOAT, OW, OH) };
  Img defExp = mkImg(VK_FORMAT_R32G32_SFLOAT, 1, 1), defReact = mkImg(VK_FORMAT_R8_UNORM, 1, 1);
  Img color = mkImg(VK_FORMAT_R32G32B32A32_SFLOAT, W, H, 1, false), depth = mkImg(VK_FORMAT_R32_SFLOAT, W, H, 1, false);
  Img motion = mkImg(VK_FORMAT_R32G32_SFLOAT, W, H, 1, false), out = mkImg(VK_FORMAT_R16G16B16A16_SFLOAT, OW, OH);   // = NFSU_CreateStorageTexture
  Img* all[] = { &dilMv, &dilDepth, &recDepth, &inter, &luma[0], &luma[1], &atom, &finfo, &spd, &far1, &shade, &dreact, &locks,
                 &acc[0], &acc[1], &lh[0], &lh[1], &up[0], &up[1], &defExp, &defReact, &color, &depth, &motion, &out };

  Buf upload = mkBuf(W * H * 16 + W * H * 4 + W * H * 8, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  Buf readback = mkBuf(OW * OH * 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  Buf cb = mkBuf(sizeof(FsrConstants), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT), cbSpd = mkBuf(sizeof(SpdC), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
  Buf cbRcas = mkBuf(16, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

  VkDescriptorPoolSize ps[3] = { { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1024 }, { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024 }, { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 256 } };
  VkDescriptorPoolCreateInfo dpi = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO }; dpi.maxSets = 128; dpi.poolSizeCount = 3; dpi.pPoolSizes = ps;
  dpi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  VkDescriptorPool pool; CK(vkCreateDescriptorPool(dev, &dpi, nullptr, &pool));
  VkCommandPoolCreateInfo cpi = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO }; cpi.queueFamilyIndex = 0; cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  VkCommandPool cpool; CK(vkCreateCommandPool(dev, &cpi, nullptr, &cpool));
  VkCommandBufferAllocateInfo cai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO }; cai.commandPool = cpool; cai.commandBufferCount = 1;
  VkCommandBuffer cmd; CK(vkAllocateCommandBuffers(dev, &cai, &cmd));

  auto barrier = [&]() {
    VkMemoryBarrier b = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_UNIFORM_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
  };
  std::vector<VkDescriptorSet> sets;
  struct D { int kind; VkImageView v; VkBuffer b; };              // kind 0 sampled, 1 storage, 2 ubo
  auto run = [&](Pipe& p, std::vector<D> d, uint32_t x, uint32_t y) {
    if (d.size() != p.n) { std::printf("binding count mismatch %zu vs %u\n", d.size(), p.n); std::exit(1); }
    VkDescriptorSetAllocateInfo ai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO }; ai.descriptorPool = pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &p.dsl;
    VkDescriptorSet set; CK(vkAllocateDescriptorSets(dev, &ai, &set)); sets.push_back(set);
    std::vector<VkWriteDescriptorSet> w(p.n); std::vector<VkDescriptorImageInfo> ii(p.n); std::vector<VkDescriptorBufferInfo> bi(p.n);
    for (uint32_t i = 0; i < p.n; i++) {
      VkDescriptorType t = d[i].kind == 0 ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : d[i].kind == 1 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      if (t != p.types[i]) { std::printf("binding %u type mismatch\n", i); std::exit(1); }
      w[i] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, i, 0, 1, t, nullptr, nullptr, nullptr };
      if (d[i].kind == 2) { bi[i] = { d[i].b, 0, VK_WHOLE_SIZE }; w[i].pBufferInfo = &bi[i]; }
      else { ii[i] = { VK_NULL_HANDLE, d[i].v, VK_IMAGE_LAYOUT_GENERAL }; w[i].pImageInfo = &ii[i]; }
    }
    vkUpdateDescriptorSets(dev, p.n, w.data(), 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.p);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pl, 0, 1, &set, 0, nullptr);
    vkCmdDispatch(cmd, x, y, 1);
    barrier();
  };
  auto Sv = [](const VkImageView& v) { return D { 0, v, VK_NULL_HANDLE }; };
  auto Uv = [](const VkImageView& v) { return D { 1, v, VK_NULL_HANDLE }; };
  auto Bf = [](const Buf& b) { return D { 2, VK_NULL_HANDLE, b.buf }; };
  auto clear = [&](Img& im, float r, float g, float b, float a) {
    VkClearColorValue v; v.float32[0] = r; v.float32[1] = g; v.float32[2] = b; v.float32[3] = a;
    VkImageSubresourceRange rg = { VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, 1 };
    vkCmdClearColorImage(cmd, im.img, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &rg);
  };

  FsrConstants k = {}; k.velocityFactor = 1; k.reactivenessScale = 1; k.shadingChangeScale = 1; k.accumulationAddedPerFrame = 1.0f / 3; k.minDisocclusionAccumulation = -1.0f / 3;
  // NFSU's own projection (logged by the mod): infinite far, near 0.5, y flipped -> as d3d9_nfsu_fsr.cpp
  const float P00 = 1.6271f, P11 = std::fabs(-2.7910f), P22 = 1.0f + 1.1920929e-7f, P32 = -0.5f;
  double rmseIn = 0, rmseOut = 0;
  for (int fr = 0; fr < frames; fr++) {
    bool odd = fr & 1, reset = fr == 0;
    float jx = halton((fr % phases) + 1, 2) - 0.5f, jy = halton((fr % phases) + 1, 3) - 0.5f;   // FSR jitter (render pixels)
    float* col = (float*)upload.map; float* dep = col + W * H * 4; float* mv = dep + W * H;
    for (uint32_t y = 0; y < H; y++) for (uint32_t x = 0; x < W; x++) {
      // projection jitter (2jx/w, -2jy/h) moves geometry by (+jx, +jy) px: the pixel centre sees the scene at centre - jitter
      int obj; shadePx((x + 0.5f - jsign * jx) * ratio, (y + 0.5f - jsign * jy) * ratio, fr, vel, col + (y * W + x) * 4, &obj);
      dep[y * W + x] = obj ? 0.9f : 0.98f;
      mv[(y * W + x) * 2 + 0] = (obj && g_objMode == 1 ? g_ovel / OW : -vel / OW) * mvmul; mv[(y * W + x) * 2 + 1] = 0.0f;      // UV, previous - current
    }
    k.previousFrameJitterOffset[0] = k.jitterOffset[0]; k.previousFrameJitterOffset[1] = k.jitterOffset[1];
    k.jitterOffset[0] = jx; k.jitterOffset[1] = jy;
    for (int i = 0; i < 2; i++) { k.previousFrameRenderSize[i] = k.renderSize[i]; k.previousFrameUpscaleSize[i] = k.upscaleSize[i]; }
    k.renderSize[0] = k.maxRenderSize[0] = W; k.upscaleSize[0] = k.maxUpscaleSize[0] = OW;
    k.renderSize[1] = k.maxRenderSize[1] = H; k.upscaleSize[1] = k.maxUpscaleSize[1] = OH;
    k.downscaleFactor[0] = float(W) / OW; k.downscaleFactor[1] = float(H) / OH;
    k.deviceToViewDepth[0] = P22; k.deviceToViewDepth[1] = P32; k.deviceToViewDepth[2] = 1 / P00; k.deviceToViewDepth[3] = 1 / P11;
    k.tanHalfFOV = 1 / P00; k.viewSpaceToMetersFactor = 1; k.motionVectorScale[0] = float(W) / W; k.motionVectorScale[1] = float(H) / H;
    k.jitterPhaseCount = float(phases); k.deltaTime = 1.0f / 60; k.deltaPreExposure = 1; k.frameIndex = reset ? 0 : k.frameIndex + 1;
    std::memcpy(cb.map, &k, sizeof k);
    SpdC s = {}; s.numWorkGroups = ((W - 1) / 64 + 1) * ((H - 1) / 64 + 1); s.mips = uint32_t(std::floor(std::log2(float(W > H ? W : H))));
    s.renderSize[0] = W; s.renderSize[1] = H; std::memcpy(cbSpd.map, &s, sizeof s);
    { float sh = std::exp2(-((-2.0f * (sharp < 0 ? 0 : sharp)) + 2.0f)); uint32_t* r = (uint32_t*)cbRcas.map; std::memcpy(r, &sh, 4);
      // packHalf2x16(sh, sh) via a tiny converter (normal range only)
      uint32_t x; std::memcpy(&x, &sh, 4); uint32_t hv = ((x >> 16) & 0x8000u) | (((((x >> 23) & 0xff) - 112) << 10) & 0x7c00u) | ((x >> 13) & 0x3ffu);
      r[1] = hv | (hv << 16); r[2] = r[3] = 0; }

    CK(vkResetCommandBuffer(cmd, 0));
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO }; CK(vkBeginCommandBuffer(cmd, &bi));
    if (fr == 0) {                                                  // UNDEFINED -> GENERAL, zero (DXVK initImage)
      std::vector<VkImageMemoryBarrier> ib;
      for (Img* im : all) { VkImageMemoryBarrier b = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER }; b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_GENERAL; b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = im->img; b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, 1 }; ib.push_back(b); }
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, uint32_t(ib.size()), ib.data());
      for (Img* im : all) clear(*im, 0, 0, 0, 0);
    }
    barrier();
    VkBufferImageCopy c = {}; c.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }; c.imageExtent = { W, H, 1 };
    c.bufferOffset = 0; vkCmdCopyBufferToImage(cmd, upload.buf, color.img, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    c.bufferOffset = W * H * 16; vkCmdCopyBufferToImage(cmd, upload.buf, depth.img, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    c.bufferOffset = W * H * 20; vkCmdCopyBufferToImage(cmd, upload.buf, motion.img, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    Img &accSrv = acc[odd ? 1 : 0], &accUav = acc[odd ? 0 : 1], &upSrv = up[odd ? 1 : 0], &upUav = up[odd ? 0 : 1];
    Img &lhSrv = lh[odd ? 1 : 0], &lhUav = lh[odd ? 0 : 1], &lCur = luma[odd ? 1 : 0], &lPrev = luma[odd ? 0 : 1];
    clear(recDepth, 1, 1, 1, 1); clear(atom, 0, 0, 0, 0); clear(spd, 0, 0, 0, 0);
    if (reset) { clear(accSrv, 0, 0, 0, 0); clear(finfo, -1, 1, 0, 0); }
    barrier();
    const uint32_t gx = (W + 7) / 8, gy = (H + 7) / 8, sx = (W - 1) / 64 + 1, sy = (H - 1) / 64 + 1, ox = (OW + 7) / 8, oy = (OH + 7) / 8;
    run(pPrep, { Sv(motion.rv), Sv(depth.rv), Sv(color.rv), Uv(dilMv.sv), Uv(dilDepth.sv), Uv(recDepth.sv), Uv(inter.sv), Uv(lCur.sv), Bf(cb) }, gx, gy);
    run(pLuma, { Sv(lCur.rv), Sv(inter.rv), Uv(atom.sv), Uv(finfo.sv), Uv(spd.mip[0]), Uv(spd.mip[1]), Uv(spd.mip[2]), Uv(spd.mip[3]), Uv(spd.mip[4]), Uv(spd.mip[5]),
                 Uv(far1.sv), Bf(cb), Bf(cbSpd) }, sx, sy);
    run(pScp, { Sv(lCur.rv), Sv(lPrev.rv), Sv(dilMv.rv), Sv(defExp.rv), Uv(atom.sv), Uv(spd.mip[0]), Uv(spd.mip[1]), Uv(spd.mip[2]), Uv(spd.mip[3]), Uv(spd.mip[4]),
                Uv(spd.mip[5]), Bf(cb), Bf(cbSpd) }, sx, sy);
    run(pSc, { Sv(spd.rv), Uv(shade.sv), Bf(cb) }, (W / 2 + 7) / 8, (H / 2 + 7) / 8);
    run(pReact, { Sv(recDepth.rv), Sv(dilMv.rv), Sv(dilDepth.rv), Sv(defReact.rv), Sv(defReact.rv), Sv(accSrv.rv), Sv(shade.rv), Sv(lCur.rv), Sv(defExp.rv),
                  Uv(dreact.sv), Uv(locks.sv), Uv(accUav.sv), Bf(cb) }, gx, gy);
    run(pInst, { Sv(defExp.rv), Sv(dreact.rv), Sv(dilMv.rv), Sv(finfo.rv), Sv(lhSrv.rv), Sv(far1.rv), Sv(lCur.rv), Uv(lhUav.sv), Uv(inter.sv), Bf(cb) }, gx, gy);
    run(sharp >= 0 ? pAccS : pAcc, { Sv(defExp.rv), Sv(dreact.rv), Sv(dilMv.rv), Sv(upSrv.rv), Sv(defExp.rv), Sv(far1.rv), Sv(lCur.rv), Sv(inter.rv), Sv(color.rv),
                Uv(upUav.sv), Uv(out.sv), Uv(locks.sv), Bf(cb) }, ox, oy);
    if (sharp >= 0) run(pRcas, { Sv(defExp.rv), Sv(upUav.rv), Uv(out.sv), Bf(cb), Bf(cbRcas) }, (OW + 15) / 16, (OH + 15) / 16);
    c.bufferOffset = 0; c.imageExtent = { OW, OH, 1 }; vkCmdCopyImageToBuffer(cmd, out.img, VK_IMAGE_LAYOUT_GENERAL, readback.buf, 1, &c);
    CK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO }; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    CK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE)); CK(vkQueueWaitIdle(queue));
    for (VkDescriptorSet s_ : sets) vkFreeDescriptorSets(dev, pool, 1, &s_);
    sets.clear();

    if (fr >= frames - 4) {                                        // RMSE over the last 4 frames, 8 px border excluded (display res)
      std::vector<float> ov(OW * OH * 4); const uint16_t* hp = (const uint16_t*)readback.map;
      for (uint32_t i = 0; i < OW * OH * 4; i++) { uint32_t h_ = hp[i], e_ = (h_ >> 10) & 31, m_ = h_ & 1023;
        float v_ = e_ == 0 ? std::ldexp(float(m_), -24) : e_ == 31 ? INFINITY : std::ldexp(float(m_ | 1024), int(e_) - 25); ov[i] = (h_ & 0x8000) ? -v_ : v_; }
      auto inp = [&](uint32_t x, uint32_t y, int i) {             // input at display pixel (x, y): bilinear from render res
        float u = (x + 0.5f) / ratio - 0.5f, v = (y + 0.5f) / ratio - 0.5f;
        int x0 = int(std::floor(u)), y0 = int(std::floor(v)); float fx = u - x0, fy = v - y0;
        auto at = [&](int xx, int yy) { xx = xx < 0 ? 0 : xx >= int(W) ? W - 1 : xx; yy = yy < 0 ? 0 : yy >= int(H) ? H - 1 : yy; return col[(yy * W + xx) * 4 + i]; };
        return (at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx) * (1 - fy) + (at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx) * fy;
      };
      const float* o = ov.data(); double eIn = 0, eOut = 0; int cnt = 0; bool bad = false;
      for (uint32_t y = 8; y < OH - 8; y++) for (uint32_t x = 8; x < OW - 8; x++) {
        float ref[3] = { 0, 0, 0 }, t[4];
        for (int sy_ = 0; sy_ < 8; sy_++) for (int sx_ = 0; sx_ < 8; sx_++) {
          int ob_; shadePx(x + (sx_ + 0.5f) / 8, y + (sy_ + 0.5f) / 8, fr, vel, t, &ob_);
          for (int i = 0; i < 3; i++) ref[i] += t[i] / 64;
        }
        for (int i = 0; i < 3; i++) {
          auto tm = [](float v) { return v / (1 + v); };          // compare in tonemapped space (HDR lines)
          float vi = inp(x, y, i), vo = o[(y * OW + x) * 4 + i];
          if (!std::isfinite(vo)) bad = true;
          eIn += (tm(vi) - tm(ref[i])) * (tm(vi) - tm(ref[i])); eOut += (tm(vo) - tm(ref[i])) * (tm(vo) - tm(ref[i])); cnt++;
        }
      }
      rmseIn += std::sqrt(eIn / cnt) / 4; rmseOut += std::sqrt(eOut / cnt) / 4;
      if (ppm && fr == frames - 1) {
        FILE* fp = std::fopen(ppm, "wb"); std::fprintf(fp, "P6 %u %u 255\n", OW * 3, OH);
        for (uint32_t y = 0; y < OH; y++) for (uint32_t pnl = 0; pnl < 3; pnl++) for (uint32_t x = 0; x < OW; x++) {
          float v[4] = { 0, 0, 0, 0 }, t[4];
          if (pnl == 0) for (int i = 0; i < 3; i++) v[i] = inp(x, y, i);
          if (pnl == 1) for (int i = 0; i < 3; i++) v[i] = o[(y * OW + x) * 4 + i];
          if (pnl == 2) for (int sy_ = 0; sy_ < 8; sy_++) for (int sx_ = 0; sx_ < 8; sx_++) {
            int ob_; shadePx(x + (sx_ + 0.5f) / 8, y + (sy_ + 0.5f) / 8, fr, vel, t, &ob_); for (int i = 0; i < 3; i++) v[i] += t[i] / 64; }
          for (int i = 0; i < 3; i++) { float tv = v[i] / (1 + v[i]); std::fputc(int(255.0f * std::sqrt(tv < 0 ? 0 : tv > 1 ? 1 : tv) + 0.5f), fp); }
        }
        std::fclose(fp);
      }
      if (bad) std::printf("frame %d: NaN/Inf in the output\n", fr);
    }
  }
  std::printf("render %ux%u -> display %ux%u (%d jitter phases)\n", W, H, OW, OH, phases);
  std::printf("frames %d, jitter sign %+.0f, sharpness %.2f: RMSE (tonemapped) aliased input %.4f, FSR output %.4f (%s), validation errors %d\n",
    frames, jsign, sharp, rmseIn, rmseOut, rmseOut < rmseIn ? "better than input" : "NOT better", g_valErrors);
  return g_valErrors ? 2 : 0;
}
