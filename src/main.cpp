#include <array>
#include <assert.h>
#include <chrono>
#include <codecvt>
#include <comdef.h>
#include <cstdio>
#include <time.h>
#include <varargs.h>
#include <vector>
#include <windows.h>

#include <tracy/tracy.hpp>

#include <volk.h>
#include <vulkan/vk_enum_string_helper.h>
#define VMA_IMPLEMENTATION
#include <vma/vk_mem_alloc.h>
#undef VMA_IMPLEMENTATION

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

#include "slang/slang.h"
#include "slang/slang-com-ptr.h"

#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1

template<typename... Args> void _unused_variable(Args&&...) {}
#define UNUSED_VARIABLE(...) _unused_variable(__VA_ARGS__)

template<typename T, size_t N>
inline constexpr size_t ArrayCount(const T (&)[N])
{
  return N;
}

// Returns a pointer to a thread-local static buffer, use with care.
static const char* GetDateString(const char* fmt, int64_t* ms_=nullptr)
{
  static thread_local char s_buf[32] = {};

  struct timespec now;
  timespec_get(&now, TIME_UTC);
  if (ms_) {
    int64_t ms = now.tv_nsec/1'000'000;
    *ms_ = ms;
  }

  struct tm* ti = localtime(&now.tv_sec);
  strftime(s_buf, ArrayCount(s_buf), fmt, ti);

  return s_buf;
}

void Log(const char* fmt, ...)
{
  static char msg[512];
  uint32_t c = 0;
  int64_t ms;
  const char* timestamp = GetDateString("%H:%M:%S", &ms);
  c += snprintf(msg + c, sizeof(msg) - c, "[%s.%03zd] ", timestamp, ms);
  va_list args;
  va_start(args, fmt);
  c += vsnprintf(msg + c, sizeof(msg) - c, fmt, args);
  va_end(args);
  printf("%s\n", msg);
  TracyMessage(msg, c);
  fflush(stdout);
}

bool check_hr(HRESULT hr)
{
  if (hr < 0) {
    _com_error err(hr);
    Log("COM error (HRESULT %d): %s", hr, err.ErrorMessage());
    return false;
  }
  return true;
}

#define CONCAT_(a, b) a##b
#define CONCAT(a, b) CONCAT_(a, b)
#define assert_hr_(name, exp) HRESULT name = exp; UNUSED_VARIABLE(name); assert(check_hr(name))
#define assert_hr(exp) assert_hr_(CONCAT(___hr, __COUNTER__), exp)

bool check_vk(VkResult result)
{
  if (result != VK_SUCCESS) {
    Log("Vulkan error (code %d): %s", result, string_VkResult(result));
    return false;
  }
  return true;
}

#define assert_vk_(name, exp) VkResult name = exp; UNUSED_VARIABLE(name); assert(check_vk(name))
#define assert_vk(exp) assert_vk_(CONCAT(___vk, __COUNTER__), exp)

static void Spinloop(double ms) {
  ZoneScoped;

  auto t1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> dur{ms};
  std::chrono::duration<double, std::milli> elapsed;
  do {
    elapsed = std::chrono::high_resolution_clock::now() - t1;
  } while (elapsed < dur);
}

constexpr uint32_t maxFramesInFlight = 2;

struct VulkanState
{
  VkInstance instance = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  VmaAllocator allocator = VK_NULL_HANDLE;
  VkCommandPool commandPool = VK_NULL_HANDLE;
  VkShaderModule shaderModule{};
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkDeviceSize vDataSize = 0;
  uint32_t numIndices = 0;
  VkBuffer vBuffer = VK_NULL_HANDLE; // Contains vertex and index data
  VmaAllocation vBufferAlloc = VK_NULL_HANDLE;
};

static VulkanState vk{};

struct ShaderDataBuffer
{
  VmaAllocation allocation = VK_NULL_HANDLE;
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceAddress deviceAddress = 0;
  void* mapped = nullptr;
};

struct Window
{
  HWND handle = NULL;
  int width = 0;
  int height = 0;

  VkSurfaceKHR surface = VK_NULL_HANDLE;
  VkSwapchainKHR swapchain = VK_NULL_HANDLE;
  std::vector<VkImage> swapchainImages{};
  std::vector<VkImageView> swapchainImageViews{};
  std::vector<VkSemaphore> renderSemaphores{};
  std::array<VkSemaphore, maxFramesInFlight> acquireSemaphores{};
  std::array<VkFence, maxFramesInFlight> acquireFences{};
  std::array<ShaderDataBuffer, maxFramesInFlight> shaderDataBuffers{};
  std::array<VkFence, maxFramesInFlight> renderFences{};
  std::array<VkCommandBuffer, maxFramesInFlight> commandBuffers{};
  std::array<VkSemaphore, maxFramesInFlight> lowLatencySemaphores{};
};

static std::vector<Window> windows;

static bool s_run = true;
static bool s_advance = true;
static bool s_advanceOnce = false;
static int s_frameCount = 0;

static LRESULT WindowProc(HWND handle, UINT message, WPARAM wparam, LPARAM lparam)
{
  ZoneScoped;

  switch (message) {
    case WM_NCCREATE:{
      auto cs = reinterpret_cast<CREATESTRUCT*>(lparam);
      auto windowIdx = (int)((uint64_t)cs->lpCreateParams);
      Log("old handle: %x, new handle %x", windows[windowIdx].handle, handle);
      windows[windowIdx].handle = handle;
    } break;

    case WM_SIZE: {
      if (wparam != SIZE_MINIMIZED) {
        for (Window& window : windows) {
          if (window.handle != handle) continue;
          window.width = LOWORD(lparam);
          window.height = HIWORD(lparam);
        }
      }
    } break;

    case WM_DPICHANGED: {
      RECT* rect = reinterpret_cast<RECT*>(lparam);
      int flags = SWP_NOZORDER|SWP_NOACTIVATE;
      ::SetWindowPos(handle, NULL, rect->left, rect->top, rect->right - rect->left, rect->bottom - rect->top, flags);
    } break;
    case WM_DESTROY:
    case WM_NCDESTROY: {
      ::DestroyWindow(handle);
      s_run = false;
    } break;

    case WM_SYSKEYDOWN:
    case WM_SYSKEYUP:
    case WM_KEYDOWN:
    case WM_KEYUP: {
      // Note: This doesn't account for non-standard keyboard layouts, or probably most international keyboards.
      bool pressed = ((lparam & (1 << 31)) == 0);
      if ((wparam == VK_ESCAPE) && pressed) {
        s_run = false; // goodbye
      }

      if ((wparam == VK_SPACE) && pressed) {
        s_advance = !s_advance;
      }

      if ((wparam == VK_RIGHT) && pressed) {
        s_advanceOnce = true;
      }
    } break;
  }

  LRESULT result = ::DefWindowProc(handle, message, wparam, lparam);
  return result;
}

static const char* RegisterWindowClass()
{
  ZoneScoped;

  WNDCLASSEX wndclass = {};
  wndclass.cbSize = sizeof(wndclass);
  wndclass.style = CS_OWNDC|CS_VREDRAW|CS_HREDRAW;
  wndclass.lpfnWndProc = WindowProc;
  wndclass.cbClsExtra = 0;
  wndclass.cbWndExtra = 0;
  wndclass.hInstance = ::GetModuleHandle(NULL);
  wndclass.hIcon = NULL;
  wndclass.hCursor = ::LoadCursor(NULL, IDC_ARROW);
  wndclass.hbrBackground = NULL;
  wndclass.lpszMenuName = NULL;
  wndclass.lpszClassName = "DualOutputSyncTestWindowClassName";
  wndclass.hIconSm = NULL;

  if (::RegisterClassEx(&wndclass) == 0) {
    DWORD error = ::GetLastError();
    Log("[Window] Failed to register win32 window class: %lu.", error);
    return nullptr;
  }

  return wndclass.lpszClassName;
}

static bool createWindow(const char* className, int idx, std::array<long, 4> rect)
{
  ZoneScoped;

  if (idx >= windows.size()) {
    windows.resize(idx+1);
  }
  windows[idx] = Window{};

  // WS_POPUP for borderless.
  DWORD style = WS_CLIPSIBLINGS|WS_CLIPCHILDREN|WS_VISIBLE|WS_SYSMENU|WS_MINIMIZEBOX|WS_POPUP;
  DWORD exstyle = WS_EX_APPWINDOW;
  HINSTANCE module = ::GetModuleHandle(NULL);
  HWND windowHandle = ::CreateWindowEx(exstyle, className, "Dual Output Sync Tester", style,
                                       rect[0], rect[1], rect[2], rect[3],
                                       NULL, NULL, module, (void*)((uint64_t)idx));

  Log("Create window handle: %x (idx %d)", windowHandle, idx);
  if (!windowHandle) {
    Log("Window creation failed: %lu.", ::GetLastError());
    return false;
  }

  // windowHandle gets stored in windows[idx] from the WindowProc callback.

  HDC hdc = ::GetDC(windowHandle);
  PIXELFORMATDESCRIPTOR pfd{};
  pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
  pfd.nVersion = 1;
  pfd.dwFlags = PFD_DRAW_TO_WINDOW;
  pfd.iPixelType = PFD_TYPE_RGBA;
  pfd.cColorBits = 24;
  pfd.cAlphaBits = 8;
  pfd.cDepthBits = 0;
  pfd.cStencilBits = 0;
  pfd.cAccumBits = 0;
  pfd.cAuxBuffers = 0;
  pfd.iLayerType = PFD_MAIN_PLANE;

  int formatIdx = ::ChoosePixelFormat(hdc, &pfd);
  // TODO: Assuming we got what we wanted for now.
  ::SetPixelFormat(hdc, formatIdx, &pfd);

  return true;
}

static void destroyWindow(Window& window)
{
  ::DestroyWindow(window.handle);
}

struct Vertex
{
  glm::vec2 pos;
  glm::vec2 uv;
};

struct ShaderData
{
  glm::mat4 rectXform{};
  uint64_t maxMetaLoop = 1;
};

static constexpr char kShader[] = R"shader(
struct VSInput {
  float2 pos;
  float2 uv;
};

struct ShaderData {
  // This should only be a 3x3 matrix since we only need a 2D xy transform.
  // However Vulkan doesn't like the 12-byte alignment of the elements of a 3x3 matrix, so we
  // go with a 4x4 here.
  float4x4 rectXform;
  // Using uint64_t here to avoid potential issues around member alignment/size.
  uint64_t maxMetaLoop;
};

struct VSOutput {
  float4 pos : SV_POSITION;
  float2 uv;
};

[shader("vertex")]
VSOutput main(VSInput input, uniform ShaderData* shaderData) {
  VSOutput output;
  float4 p = float4(input.pos, 0.0, 1.0);
  p = mul(shaderData->rectXform, p);
  output.pos = float4(p.xy, 0.0, 1.0);
  output.uv = input.uv;
  return output;
}

#define MAX_BROT 200

float mandelbrot(float2 c) {
  float2 z = float2(0.0, 0.0);
  int i = 0;
  for (; i < MAX_BROT; i++) {
    z = float2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
    if (z.x*z.x + z.y*z.y > 4) break;
  }

  float grade = 0.0;
  if (i < MAX_BROT) {
    grade = float(i) / float(MAX_BROT);
  }

  return grade;
}


// Takes our seed, updates it, and returns a pseudorandom float in [0..1]
float nextRand(inout uint s) {
    s = (1664525u * s + 1013904223u);
    return float(s & 0x00FFFFFF) / float(0x01000000);
}

float2 rand2(inout uint s) {
  return float2(nextRand(s), nextRand(s));
}

[shader("fragment")]
float4 main(VSOutput input, uniform ShaderData* shaderData) {
  if (input.uv.x < 0.1 || input.uv.x > 0.9) {
    return float4(1.0, 0.0, 0.0, 1.0);
  }

  float grade = 0.0;
  uint seed = (uint)input.pos.x + (uint)input.pos.y;
  for (int i = 0; i < shaderData->maxMetaLoop; i++) {
    float2 p = input.uv + rand2(seed) / float2(1920, 1080);
    float2 c = float2(-2.5 + p.x * 3.5, -1.0 + p.y * 2.0);
    grade += mandelbrot(c);
  }

  grade = grade / float(shaderData->maxMetaLoop);

  return float4(0.0, grade, grade, 1.0);
}
)shader";

int main(int argc, const char** argv)
{
  UNUSED_VARIABLE(argc, argv);

  std::vector<HMONITOR> monitors;
  auto monitorEnumProc = [](HMONITOR monitor, HDC hdc, LPRECT rect, LPARAM param) -> int {
    UNUSED_VARIABLE(hdc, rect);
    auto ms = reinterpret_cast<std::vector<HMONITOR>*>(param);
    ms->emplace_back(monitor);
    return true;
  };
  ::EnumDisplayMonitors(NULL, NULL, monitorEnumProc, reinterpret_cast<LPARAM>(&monitors));

  std::vector<std::array<long, 4>> monitorRects;
  for (const HMONITOR& monitor : monitors) {
    MONITORINFO info{0};
    info.cbSize = sizeof(MONITORINFO);
    if(!::GetMonitorInfo(monitor, &info)) {
      Log("Failed to get monitor info for %p", monitor);
      continue;
    }

    monitorRects.emplace_back(std::array<long, 4>{
      info.rcMonitor.left, info.rcMonitor.top,
      info.rcMonitor.right-info.rcMonitor.left, info.rcMonitor.bottom-info.rcMonitor.top
    });

    Log("Monitor %zu rect: %ld, %ld, %ld, %ld", monitorRects.size() - 1,
        monitorRects.back()[0],
        monitorRects.back()[1],
        monitorRects.back()[2],
        monitorRects.back()[3]);
  }

  if (monitorRects.size() > 2) {
    Log("Only using the first two monitors.");
    monitorRects.resize(2);
  }
  monitorRects.resize(1);

  const char* className = RegisterWindowClass();
  if (!className) return EXIT_FAILURE;

  for (int i = 0; i < monitorRects.size(); i++) {
    bool success = createWindow(className, i, monitorRects[i]);
    if (!success) {
      Log("Failed to create window %d!", i);
      return EXIT_FAILURE;
    }
  }

  // Vulkan init
  // Stuff we need to do before we can create windows and associated resources.
  assert_vk(volkInitialize());

  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "Dual Output Sync Tester";
  appInfo.apiVersion = VK_API_VERSION_1_3;

  // Taken from SFML's getGraphicsRequiredInstanceExtensions() for win32
  const std::vector<const char*> instanceExtensions { VK_KHR_SURFACE_EXTENSION_NAME,
                                                      VK_KHR_WIN32_SURFACE_EXTENSION_NAME };

  VkInstanceCreateInfo instCI {};
  instCI.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instCI.pApplicationInfo = &appInfo;
  instCI.enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size());
  instCI.ppEnabledExtensionNames = instanceExtensions.data();

  assert_vk(vkCreateInstance(&instCI, nullptr, &vk.instance));
  volkLoadInstance(vk.instance);

  // Make our surfaces, because it's required for finding a presentation queue.
  for (Window& window : windows) {
    VkWin32SurfaceCreateInfoKHR surfaceCI{};
    surfaceCI.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    surfaceCI.hinstance = ::GetModuleHandle(nullptr);
    surfaceCI.hwnd = window.handle;

    vkCreateWin32SurfaceKHR(vk.instance, &surfaceCI, nullptr, &window.surface);
  }

  // Get devices
  uint32_t deviceCount = 0;
  assert_vk(vkEnumeratePhysicalDevices(vk.instance, &deviceCount, nullptr));
  std::vector<VkPhysicalDevice> devices(deviceCount);
  assert_vk(vkEnumeratePhysicalDevices(vk.instance, &deviceCount, devices.data()));

  uint32_t deviceIndex = UINT32_MAX;
  for (uint32_t i = 0; i < deviceCount; i++) {
    VkPhysicalDeviceProperties deviceProps{};
    vkGetPhysicalDeviceProperties(devices[i], &deviceProps);
    if (deviceProps.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
      deviceIndex = i;
      break;
    }
  }
  if (deviceIndex == UINT32_MAX) {
    assert(!"Couldn't find an integrated GPU device.");
    return EXIT_FAILURE;
  }

  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(devices[deviceIndex], &queueFamilyCount, nullptr);
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(devices[deviceIndex], &queueFamilyCount, queueFamilies.data());

  uint32_t queueFamily = UINT32_MAX;
  for (size_t i = 0; i < queueFamilies.size(); i++) {
    if (!(queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;

    bool supportsPresentation = true;
    for (Window& window : windows) {
      VkBool32 supportsPresentationForWindow = false;
      assert_vk(vkGetPhysicalDeviceSurfaceSupportKHR(devices[deviceIndex], static_cast<uint32_t>(i),
                                                     window.surface, &supportsPresentationForWindow));
      if (!supportsPresentationForWindow) {
        supportsPresentation = false;
        break;
      }

      // We do not check for support for low_latency2 here, because when I tried to use
      // vkPhysicalDeviceGetSurfaceCapabilities2KHR with a VkLatencySurfaceCapabilitiesNV in the input chain,
      // I didn't get any useful data out - it just didn't seem to write anything.
    }
    if (!supportsPresentation) continue;

    queueFamily = static_cast<uint32_t>(i);
    break;
  }
  if (queueFamily == UINT32_MAX) {
    Log("No viable graphics queue was found.");
    return EXIT_FAILURE;
  }

  // Create device and queue
  const float qfPriorities = 1.0f; // We'll just make a single graphics queue
  VkDeviceQueueCreateInfo queueCI{};
  queueCI.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCI.queueFamilyIndex = queueFamily;
  queueCI.queueCount = 1;
  queueCI.pQueuePriorities = &qfPriorities;

  // TODO: review what's needed of the below
  VkPhysicalDeviceVulkan12Features enabledVk12Features{};
  enabledVk12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  enabledVk12Features.descriptorIndexing = true;
  enabledVk12Features.descriptorBindingVariableDescriptorCount = true;
  enabledVk12Features.runtimeDescriptorArray = true;
  enabledVk12Features.bufferDeviceAddress = true;
  enabledVk12Features.timelineSemaphore = true;

  VkPhysicalDeviceVulkan13Features enabledVk13Features{};
  enabledVk13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
  enabledVk13Features.pNext = &enabledVk12Features;
  enabledVk13Features.synchronization2 = true;
  enabledVk13Features.dynamicRendering = true;

  VkPhysicalDevicePresentIdFeaturesKHR presentIdFeatures{};
  presentIdFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR;
  presentIdFeatures.pNext = &enabledVk13Features;
  presentIdFeatures.presentId = true;

  VkPhysicalDeviceFeatures enabledVk10Features {};
  enabledVk10Features.samplerAnisotropy = true;
  // This seems to be required because the Slang compiler outputs SPIR-V that declares the `Int64` capability.
  // I'm not quite sure what in our shader makes it declare that capability, for now this should be an easy enough workaround.
  // See https://github.com/shader-slang/slang/issues/7562 for a possibly similar issue someone had.
  enabledVk10Features.shaderInt64 = true;

  const std::vector<const char*> deviceExtensions { VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_PRESENT_ID_EXTENSION_NAME,
                                                    VK_NV_LOW_LATENCY_2_EXTENSION_NAME };

  VkDeviceCreateInfo deviceCI{};
  deviceCI.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceCI.pNext = &presentIdFeatures;
  deviceCI.queueCreateInfoCount = 1;
  deviceCI.pQueueCreateInfos = &queueCI;
  deviceCI.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
  deviceCI.ppEnabledExtensionNames = deviceExtensions.data();
  deviceCI.pEnabledFeatures = &enabledVk10Features;

  assert_vk(vkCreateDevice(devices[deviceIndex], &deviceCI, nullptr, &vk.device));

  vkGetDeviceQueue(vk.device, queueFamily, 0, &vk.queue);

  // Vulkan Memory Allocator
  VmaVulkanFunctions vkFunctions{};
  vkFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
  vkFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
  vkFunctions.vkCreateImage = vkCreateImage;

  VmaAllocatorCreateInfo allocatorCI{};
  allocatorCI.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
  allocatorCI.physicalDevice = devices[deviceIndex];
  allocatorCI.device = vk.device;
  allocatorCI.pVulkanFunctions = &vkFunctions;
  allocatorCI.instance = vk.instance;

  assert_vk(vmaCreateAllocator(&allocatorCI, &vk.allocator));

  VkCommandPoolCreateInfo commandPoolCI{};
  commandPoolCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  commandPoolCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  commandPoolCI.queueFamilyIndex = queueFamily;
  assert_vk(vkCreateCommandPool(vk.device, &commandPoolCI, nullptr, &vk.commandPool));

  {
    std::vector<Vertex> vertices;
    std::vector<uint16_t> indices;

    for (int i = 0; i < 4; i++) {
      Vertex v{};
      v.pos.x = (float)(i % 2);
      v.pos.y = (float)(i / 2);
      v.uv = v.pos;

      vertices.push_back(v);
    }

    // Assuming x+ right, y+ up, ccw winding
    indices.push_back(0);
    indices.push_back(1);
    indices.push_back(2);
    indices.push_back(1);
    indices.push_back(3);
    indices.push_back(2);

    vk.numIndices = static_cast<uint32_t>(indices.size());
    vk.vDataSize = sizeof(Vertex) * vertices.size();
    VkDeviceSize iDataSize = sizeof(uint16_t) * indices.size();

    VkBufferCreateInfo bufferCI{};
    bufferCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCI.size = static_cast<VkDeviceSize>(vk.vDataSize + iDataSize);
    bufferCI.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;

    VmaAllocationCreateInfo bufferAllocCI{};
    bufferAllocCI.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    bufferAllocCI.usage = VMA_MEMORY_USAGE_AUTO;

    assert_vk(vmaCreateBuffer(vk.allocator, &bufferCI, &bufferAllocCI, &vk.vBuffer, &vk.vBufferAlloc, nullptr));

    void* bufferPtr = nullptr;
    vmaMapMemory(vk.allocator, vk.vBufferAlloc, &bufferPtr);
    memcpy(bufferPtr, vertices.data(), vk.vDataSize);
    memcpy(((uint8_t*)bufferPtr) + vk.vDataSize, indices.data(), iDataSize);
    vmaUnmapMemory(vk.allocator, vk.vBufferAlloc);
  }

  const VkFormat swapchainImageFormat = VK_FORMAT_B8G8R8A8_SRGB;
  for (Window& window : windows) {
    VkSurfaceCapabilitiesKHR surfaceCaps{};
    assert_vk(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(devices[deviceIndex], window.surface, &surfaceCaps));

    VkSwapchainLatencyCreateInfoNV latencyCI{};
    latencyCI.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_LATENCY_CREATE_INFO_NV;
    latencyCI.latencyModeEnable = true;

    VkSwapchainCreateInfoKHR swapchainCI{};
    swapchainCI.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchainCI.pNext = &latencyCI;
    swapchainCI.surface = window.surface;
    swapchainCI.minImageCount = surfaceCaps.minImageCount;
    swapchainCI.imageFormat = swapchainImageFormat;
    swapchainCI.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    swapchainCI.imageExtent = VkExtent2D { surfaceCaps.currentExtent.width, surfaceCaps.currentExtent.height };
    swapchainCI.imageArrayLayers = 1;
    swapchainCI.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    swapchainCI.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    swapchainCI.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapchainCI.presentMode = VK_PRESENT_MODE_FIFO_KHR;

    assert_vk(vkCreateSwapchainKHR(vk.device, &swapchainCI, nullptr, &window.swapchain));

    VkLatencySleepModeInfoNV sleepModeInfo{};
    sleepModeInfo.sType = VK_STRUCTURE_TYPE_LATENCY_SLEEP_MODE_INFO_NV;
    sleepModeInfo.lowLatencyMode = true;
    sleepModeInfo.lowLatencyBoost = true;
    sleepModeInfo.minimumIntervalUs = 1000000 / 60;
    vkSetLatencySleepModeNV(vk.device, window.swapchain, &sleepModeInfo);

    uint32_t imageCount = 0;
    vkGetSwapchainImagesKHR(vk.device, window.swapchain, &imageCount, nullptr);
    window.swapchainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(vk.device, window.swapchain, &imageCount, window.swapchainImages.data());

    window.swapchainImageViews.resize(imageCount);
    for (uint32_t i = 0; i < imageCount; i++) {
      VkImageViewCreateInfo viewCI{};
      viewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      viewCI.image = window.swapchainImages[i];
      viewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
      viewCI.format = swapchainImageFormat;
      viewCI.subresourceRange = VkImageSubresourceRange{};
      viewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      viewCI.subresourceRange.levelCount = 1;
      viewCI.subresourceRange.layerCount = 1;
      assert_vk(vkCreateImageView(vk.device, &viewCI, nullptr, &window.swapchainImageViews[i]));
    }

    for (ShaderDataBuffer& sdb : window.shaderDataBuffers) {
      VkBufferCreateInfo bufferCI2{};
      bufferCI2.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
      bufferCI2.size = sizeof(ShaderData);
      bufferCI2.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

      VmaAllocationCreateInfo bufferAllocCI2{};
      bufferAllocCI2.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                             VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
      bufferAllocCI2.usage = VMA_MEMORY_USAGE_AUTO;
      assert_vk(vmaCreateBuffer(vk.allocator, &bufferCI2, &bufferAllocCI2, &sdb.buffer,
                                &sdb.allocation, nullptr));
      vmaMapMemory(vk.allocator, sdb.allocation, &sdb.mapped);

      VkBufferDeviceAddressInfo bufferBdaInfo{};
      bufferBdaInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
      bufferBdaInfo.buffer = sdb.buffer;
      sdb.deviceAddress = vkGetBufferDeviceAddress(vk.device, &bufferBdaInfo);
    }

    VkSemaphoreCreateInfo semaphoreCI{};
    semaphoreCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkSemaphoreTypeCreateInfo semaphoreTypeCI{};
    semaphoreTypeCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    semaphoreTypeCI.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo timelineSemaphoreCI{};

    timelineSemaphoreCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    timelineSemaphoreCI.pNext = &semaphoreTypeCI;

    VkFenceCreateInfo fenceCI{};
    fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (int i = 0; i < maxFramesInFlight; i++) {
      assert_vk(vkCreateFence(vk.device, &fenceCI, nullptr, &window.acquireFences[i]));
      assert_vk(vkCreateFence(vk.device, &fenceCI, nullptr, &window.renderFences[i]));
      assert_vk(vkCreateSemaphore(vk.device, &semaphoreCI, nullptr, &window.acquireSemaphores[i]));
      assert_vk(vkCreateSemaphore(vk.device, &timelineSemaphoreCI, nullptr, &window.lowLatencySemaphores[i]));
    }

    window.renderSemaphores.resize(window.swapchainImages.size());
    for (auto& semaphore : window.renderSemaphores) {
      assert_vk(vkCreateSemaphore(vk.device, &semaphoreCI, nullptr, &semaphore));
    }

    VkCommandBufferAllocateInfo cbAllocCI{};
    cbAllocCI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbAllocCI.commandPool = vk.commandPool;
    cbAllocCI.commandBufferCount = maxFramesInFlight;
    assert_vk(vkAllocateCommandBuffers(vk.device, &cbAllocCI, window.commandBuffers.data()));
  }

  {
    // Slang
    Slang::ComPtr<slang::IGlobalSession> slangGlobalSession;
    slang::createGlobalSession(slangGlobalSession.writeRef());

    slang::TargetDesc target{};
    target.format = SLANG_SPIRV;
    target.profile = slangGlobalSession->findProfile("spirv_1_4");

    slang::CompilerOptionEntry slangOption{};
    slangOption.name = slang::CompilerOptionName::EmitSpirvDirectly;
    slangOption.value.kind = slang::CompilerOptionValueKind::Int;
    slangOption.value.intValue0 = 1;

    slang::SessionDesc slangSessionDesc{};
    slangSessionDesc.targets = &target;
    slangSessionDesc.targetCount = SlangInt(1);
    slangSessionDesc.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_COLUMN_MAJOR;
    slangSessionDesc.compilerOptionEntries = &slangOption;
    slangSessionDesc.compilerOptionEntryCount = uint32_t(1);

    Slang::ComPtr<slang::ISession> slangSession;
    assert(SLANG_SUCCEEDED(slangGlobalSession->createSession(slangSessionDesc, slangSession.writeRef())));

    Slang::ComPtr<slang::IModule> slangModule;
    Slang::ComPtr<ISlangBlob> slangDiagnostics;
    slangModule = slangSession->loadModuleFromSourceString("rect", "rect.slang", kShader, slangDiagnostics.writeRef());
    if (slangDiagnostics) {
      auto diagnosticMessage = reinterpret_cast<const char*>(slangDiagnostics->getBufferPointer());
      Log("Slang compiler diagnostics:\n%s", diagnosticMessage);
      assert(!"Slang compilation failed, see log!");
    }

    Slang::ComPtr<ISlangBlob> spirv;
    slangModule->getTargetCode(0, spirv.writeRef());

    VkShaderModuleCreateInfo shaderModuleCI{};
    shaderModuleCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCI.codeSize = spirv->getBufferSize();
    shaderModuleCI.pCode = (uint32_t*)spirv->getBufferPointer();
    assert_vk(vkCreateShaderModule(vk.device, &shaderModuleCI, nullptr, &vk.shaderModule));
  }

  { // pipeline
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.size = sizeof(VkDeviceAddress);

    VkPipelineLayoutCreateInfo pipelineLayoutCI{};
    pipelineLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCI.pushConstantRangeCount = 1;
    pipelineLayoutCI.pPushConstantRanges = &pushConstantRange;

    assert_vk(vkCreatePipelineLayout(vk.device, &pipelineLayoutCI, nullptr, &vk.pipelineLayout));

    VkVertexInputBindingDescription vertexBinding{};
    vertexBinding.binding   = 0;
    vertexBinding.stride    = sizeof(Vertex);
    vertexBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::vector<VkVertexInputAttributeDescription> vertexAttributes{};
    vertexAttributes.emplace_back(VkVertexInputAttributeDescription
                                    { 0, 0, VK_FORMAT_R32G32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, pos)) });
    vertexAttributes.emplace_back(VkVertexInputAttributeDescription
                                    { 1, 0, VK_FORMAT_R32G32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, uv))  });

    VkPipelineVertexInputStateCreateInfo vertexInputState{};
    vertexInputState.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputState.vertexBindingDescriptionCount   = 1;
    vertexInputState.pVertexBindingDescriptions      = &vertexBinding;
    vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexAttributes.size());
    vertexInputState.pVertexAttributeDescriptions    = vertexAttributes.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssemblyState{};
    inputAssemblyState.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    std::vector<VkPipelineShaderStageCreateInfo> shaderStages{};
    VkPipelineShaderStageCreateInfo stageCI{};
    stageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageCI.stage = VK_SHADER_STAGE_VERTEX_BIT;
    stageCI.module = vk.shaderModule;
    stageCI.pName = "main";
    shaderStages.push_back(stageCI);
    stageCI.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shaderStages.push_back(stageCI);

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    std::vector<VkDynamicState> dynamicStates{ VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkPipelineRenderingCreateInfo renderingCI{};
    renderingCI.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingCI.colorAttachmentCount = 1;
    renderingCI.pColorAttachmentFormats = &swapchainImageFormat;
    // No depth attachment, skipping that since we won't need it.

    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo colorBlendState{};
    colorBlendState.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlendState.attachmentCount = 1;
    colorBlendState.pAttachments = &blendAttachment;

    VkPipelineRasterizationStateCreateInfo rasterizationState{};
    rasterizationState.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizationState.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisampleState{};
    multisampleState.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkGraphicsPipelineCreateInfo pipelineCI{};
    pipelineCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineCI.pNext = &renderingCI;
    pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineCI.pStages = shaderStages.data();
    pipelineCI.pVertexInputState = &vertexInputState;
    pipelineCI.pInputAssemblyState = &inputAssemblyState;
    pipelineCI.pViewportState = &viewportState;
    pipelineCI.pRasterizationState = &rasterizationState;
    pipelineCI.pMultisampleState = &multisampleState;
    pipelineCI.pColorBlendState = &colorBlendState;
    pipelineCI.pDynamicState = &dynamicState;
    pipelineCI.layout = vk.pipelineLayout;

    assert_vk(vkCreateGraphicsPipelines(vk.device, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &vk.pipeline));
  }

  auto PumpMessages = []() {
    ZoneScopedN("PumpMessages");
    for (int i = 0; i < windows.size(); i++) {
      Log("  Pumping messages for window %d", i);
      MSG message;
      while (::PeekMessage(&message, windows[i].handle, 0, 0, PM_REMOVE)) {
        ::TranslateMessage(&message);
        ::DispatchMessage(&message);
      }
    }
  };

  uint32_t skipCounter = 0;
  constexpr uint32_t numRenderFrames = 5;
  constexpr uint32_t frameCyclePeriod = 12;

  int linePos = 0;
  auto frameStart = std::chrono::high_resolution_clock::now();
  while (s_run) {
    FrameMark;

    if (skipCounter % frameCyclePeriod >= numRenderFrames) {
      ZoneScopedN("non-render frame");
      Spinloop(16.6);
      PumpMessages();

      skipCounter++;
      continue;
    }

    auto frameEnd = frameStart;
    frameStart = std::chrono::high_resolution_clock::now();
    Log("Frame took %.2fms", std::chrono::duration<double, std::milli>(frameStart - frameEnd).count());
    Log("Beginning local frame %d", s_frameCount);

    uint32_t resourceIdx = s_frameCount % maxFramesInFlight;
    {
      ZoneScopedN("wait on fences");
      // Wait on frame fences of all windows
      std::vector<VkFence> waitFences(windows.size());
      for (size_t i = 0; i < windows.size(); i++) {
        waitFences[i] = windows[i].renderFences[resourceIdx];
      }
      assert_vk(
        vkWaitForFences(vk.device, static_cast<uint32_t>(waitFences.size()), waitFences.data(), true, UINT64_MAX));
      assert_vk(vkResetFences(vk.device, static_cast<uint32_t>(waitFences.size()), waitFences.data()));
    }

    {
      ZoneScopedN("Wait on low latency semaphores");

      std::vector<VkSemaphore> waitSemaphores{windows.size()};
      std::vector<uint64_t> semaphoreValues{windows.size()};
      for (size_t i = 0; i < windows.size(); i++) {
        Window& window = windows[i];

        VkLatencySleepInfoNV sleepInfo{};
        sleepInfo.sType = VK_STRUCTURE_TYPE_LATENCY_SLEEP_INFO_NV;
        sleepInfo.signalSemaphore = window.lowLatencySemaphores[resourceIdx];
        sleepInfo.value = s_frameCount;
        vkLatencySleepNV(vk.device, window.swapchain, &sleepInfo);
        waitSemaphores[i] = sleepInfo.signalSemaphore;
        semaphoreValues[i] = s_frameCount;
      }

      VkSemaphoreWaitInfo waitInfo{};
      waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
      waitInfo.semaphoreCount = static_cast<uint32_t>(waitSemaphores.size());
      waitInfo.pSemaphores = waitSemaphores.data();
      waitInfo.pValues = semaphoreValues.data();

      vkWaitSemaphores(vk.device, &waitInfo, 1000000000);
    }

    uint64_t presentId = s_frameCount+1; // +1 because we need it to be non-0. PresentId 0 means "ID not set".

    std::vector<uint32_t> imageIdxs(windows.size());
    for (size_t i = 0; i < windows.size(); i++) {
      Window &window = windows[i];
      ZoneScopedN("draw loop");
      ZoneTextF("Window %d", i);

      {
        ZoneScopedN("Wait on acquire fence");
        // Ensure that the semaphore we're about to re-use is unsignaled and free of dependencies,
        // by waiting for the fence that was signaled at the same time as the semaphore was signaled (as I understand it.)
        assert_vk(vkWaitForFences(vk.device, 1, &window.acquireFences[resourceIdx], true, UINT64_MAX));
        assert_vk(vkResetFences(vk.device, 1, &window.acquireFences[resourceIdx]));
      }

      uint32_t imageIdx;
      assert_vk(vkAcquireNextImageKHR(vk.device, window.swapchain, UINT64_MAX,
                            window.acquireSemaphores[resourceIdx], window.acquireFences[resourceIdx], &imageIdx));
      imageIdxs[i] = imageIdx;

      float rect[4] = {
        (float) window.width / 2.0f, (float) (window.height - linePos % window.height),
        (float) window.width, 100
      };
      rect[1] -= rect[3] / 2;

      ShaderData shaderData;
      shaderData.rectXform = glm::mat4(1.0f);
      shaderData.rectXform[0][0] = 2.0f * rect[2] / (float) window.width;
      shaderData.rectXform[1][1] = 2.0f * rect[3] / (float) window.height;
      shaderData.rectXform[3][0] =
        -rect[2] / (float) window.width + (rect[0] - (float) window.width / 2) / ((float) window.width / 2);
      shaderData.rectXform[3][1] =
        -rect[3] / (float) window.height + (rect[1] - (float) window.height / 2) / ((float) window.height / 2);
      shaderData.maxMetaLoop = 1;
      memcpy(window.shaderDataBuffers[resourceIdx].mapped, &shaderData, sizeof(shaderData));

      {
        ZoneScopedN("buffer prepare & submit");
        ZoneTextF("Resource idx: %d", resourceIdx);
        VkCommandBuffer cb = window.commandBuffers[resourceIdx];
        vkResetCommandBuffer(cb, 0);

        VkCommandBufferBeginInfo cbBI{};
        cbBI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cbBI.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cb, &cbBI);

        VkImageMemoryBarrier2 outputBarrier{};
        outputBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        outputBarrier.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        outputBarrier.srcAccessMask = 0;
        outputBarrier.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        outputBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        outputBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        outputBarrier.newLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
        outputBarrier.image = window.swapchainImages[imageIdx];
        outputBarrier.subresourceRange = VkImageSubresourceRange{};
        outputBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        outputBarrier.subresourceRange.levelCount = 1;
        outputBarrier.subresourceRange.layerCount = 1;

        VkDependencyInfo outputBarrierDI{};
        outputBarrierDI.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        outputBarrierDI.imageMemoryBarrierCount = 1;
        outputBarrierDI.pImageMemoryBarriers = &outputBarrier;
        vkCmdPipelineBarrier2(cb, &outputBarrierDI);

        VkRenderingAttachmentInfo colorAttachmentInfo{};
        colorAttachmentInfo.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAttachmentInfo.imageView = window.swapchainImageViews[imageIdx];
        colorAttachmentInfo.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
        colorAttachmentInfo.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachmentInfo.clearValue = VkClearValue{};
        colorAttachmentInfo.clearValue.color = VkClearColorValue{0.0f, 0.0f, 0.0f, 1.0f};

        VkRenderingInfo renderingInfo{};
        renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderingInfo.renderArea = VkRect2D{};
        renderingInfo.renderArea.extent = VkExtent2D{(uint32_t) window.width, (uint32_t) window.height};
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachmentInfo;
        vkCmdBeginRendering(cb, &renderingInfo);

        VkViewport vp{};
        vp.width = static_cast<float>(window.width);
        vp.height = static_cast<float>(window.height);
        vp.minDepth = 0.0f;
        vp.maxDepth = 1.0f;
        vkCmdSetViewport(cb, 0, 1, &vp);
        VkRect2D scissor{};
        scissor.extent = VkExtent2D{(uint32_t) window.width, (uint32_t) window.height};
        vkCmdSetScissor(cb, 0, 1, &scissor);

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, vk.pipeline);
        VkDeviceSize vOffset = 0;
        vkCmdBindVertexBuffers(cb, 0, 1, &vk.vBuffer, &vOffset);
        vkCmdBindIndexBuffer(cb, vk.vBuffer, vk.vDataSize, VK_INDEX_TYPE_UINT16);

        vkCmdPushConstants(cb, vk.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(VkDeviceAddress), &window.shaderDataBuffers[resourceIdx].deviceAddress);

        vkCmdDrawIndexed(cb, vk.numIndices, 1, 0, 0, 0);
        vkCmdEndRendering(cb);

        VkImageMemoryBarrier2 barrierPresent{};
        barrierPresent.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrierPresent.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        barrierPresent.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrierPresent.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        barrierPresent.dstAccessMask = 0;
        barrierPresent.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrierPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrierPresent.image = window.swapchainImages[imageIdx];
        barrierPresent.subresourceRange = VkImageSubresourceRange{};
        barrierPresent.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrierPresent.subresourceRange.levelCount = 1;
        barrierPresent.subresourceRange.layerCount = 1;

        VkDependencyInfo barrierPresentDI{};
        barrierPresentDI.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        barrierPresentDI.imageMemoryBarrierCount = 1;
        barrierPresentDI.pImageMemoryBarriers = &barrierPresent;
        vkCmdPipelineBarrier2(cb, &barrierPresentDI);

        vkEndCommandBuffer(cb);

        VkLatencySubmissionPresentIdNV latencyInfo{};
        latencyInfo.sType = VK_STRUCTURE_TYPE_LATENCY_SUBMISSION_PRESENT_ID_NV;
        latencyInfo.presentID = presentId;

        VkPipelineStageFlags waitStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.pNext = &latencyInfo;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &window.acquireSemaphores[resourceIdx];
        submitInfo.pWaitDstStageMask = &waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cb;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &window.renderSemaphores[imageIdx];
        assert_vk(vkQueueSubmit(vk.queue, 1, &submitInfo, window.renderFences[resourceIdx]));
      }
    }

    {
      ZoneScopedN("queue present");
      {
        char imgIdxList[128];
        int c = 0;
        c += snprintf(imgIdxList, ArrayCount(imgIdxList), "Image idxs: ");
        for (size_t i = 0; i < windows.size(); i++) {
          c += snprintf(imgIdxList + c, ArrayCount(imgIdxList) - c, "%d%s", imageIdxs[i], i < (windows.size() - 1) ? ", " : "\n");
        }
        ZoneTextF("%s", imgIdxList);
      }

      uint32_t swapchainCount = static_cast<uint32_t>(windows.size());
      std::vector<VkSemaphore> semaphores(swapchainCount);
      std::vector<VkSwapchainKHR> swapchains(swapchainCount);
      std::vector<uint64_t> presentIds(swapchainCount);
      for (size_t i = 0; i < swapchainCount; i++) {
        semaphores[i] = windows[i].renderSemaphores[imageIdxs[i]];
        swapchains[i] = windows[i].swapchain;
        // Generate unique present IDs per swapchain, because low_latency2 does not take a swapchain to identify which one to wait on.
        presentIds[i] = presentId;
      }

      VkPresentIdKHR presentIdInfo{};
      presentIdInfo.sType = VK_STRUCTURE_TYPE_PRESENT_ID_KHR;
      presentIdInfo.swapchainCount = swapchainCount;
      presentIdInfo.pPresentIds = presentIds.data();

      VkPresentInfoKHR presentInfo{};
      presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
      presentInfo.pNext = &presentIdInfo;
      presentInfo.waitSemaphoreCount = static_cast<uint32_t>(semaphores.size());
      presentInfo.pWaitSemaphores = semaphores.data();
      presentInfo.swapchainCount = swapchainCount;
      presentInfo.pSwapchains = swapchains.data();
      presentInfo.pImageIndices = imageIdxs.data();
      assert_vk(vkQueuePresentKHR(vk.queue, &presentInfo));
    }

    PumpMessages();

    if (s_advance || s_advanceOnce) {
      linePos++;
    }
    s_advanceOnce = false;

    s_frameCount++;
    skipCounter++;
  }

  assert_vk(vkDeviceWaitIdle(vk.device));
  for (Window& window : windows) {
    for (auto i = 0; i < maxFramesInFlight; i++) {
      vkDestroyFence(vk.device, window.acquireFences[i], nullptr);
      vkDestroyFence(vk.device, window.renderFences[i], nullptr);
      vkDestroySemaphore(vk.device, window.acquireSemaphores[i], nullptr);
      vkDestroySemaphore(vk.device, window.renderSemaphores[i], nullptr);
      vkDestroySemaphore(vk.device, window.lowLatencySemaphores[i], nullptr);
      vmaUnmapMemory(vk.allocator, window.shaderDataBuffers[i].allocation);
      vmaDestroyBuffer(vk.allocator, window.shaderDataBuffers[i].buffer, window.shaderDataBuffers[i].allocation);
    }
    for (VkImageView& view : window.swapchainImageViews) {
      vkDestroyImageView(vk.device, view, nullptr);
    }

    vkDestroySwapchainKHR(vk.device, window.swapchain, nullptr);
    vkDestroySurfaceKHR(vk.instance, window.surface, nullptr);
  }

  vmaDestroyBuffer(vk.allocator, vk.vBuffer, vk.vBufferAlloc);
  vkDestroyPipelineLayout(vk.device, vk.pipelineLayout, nullptr);
  vkDestroyPipeline(vk.device, vk.pipeline, nullptr);
  vkDestroyCommandPool(vk.device, vk.commandPool, nullptr);
  vkDestroyShaderModule(vk.device, vk.shaderModule, nullptr);
  vmaDestroyAllocator(vk.allocator);
  vkDestroyDevice(vk.device, nullptr);
  vkDestroyInstance(vk.instance, nullptr);

  for (int i = 0; i < windows.size(); i++) {
    destroyWindow(windows[i]);
  }

  return EXIT_SUCCESS;
}
