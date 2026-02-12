#define WIN32_LEAN_AND_MEAN
#define NOMINMAX 1

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

#include <dxgi1_4.h>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <d3dcompiler.h>
#include <d3dx12/d3dx12.h>
#include <DirectXMath.h>
#include "ComUtils.h"

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
constexpr uint32_t numBackbuffers = maxFramesInFlight;

struct Vertex
{
    DirectX::XMFLOAT2 pos;
    DirectX::XMFLOAT2 uv;
};

struct ShaderData {
    float rectXform_r1[3]; [[maybe_unused]] float _pad1;
    float rectXform_r2[3]; [[maybe_unused]] float _pad2;
    float rectXform_r3[3];
    uint32_t metaLoopCount;
    [[maybe_unused]] float _padding[64 - 12]; // Pad to 256-byte alignment
};
static_assert(sizeof(ShaderData) == 256); // Check for exact size to make sure we don't waste space on extra padding.

struct DxState
{
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> commandQueue;

    UINT rtvDescriptorSize;
    UINT cbvDescriptorSize;

    ComPtr<ID3D12DescriptorHeap> rtvHeap;
    ComPtr<ID3D12DescriptorHeap> cbvHeap;
    ComPtr<ID3D12RootSignature> rootSignature;
    ComPtr<ID3D12PipelineState> pipelineState;
    ComPtr<ID3D12Resource> vertexBuffer;
    D3D12_VERTEX_BUFFER_VIEW vertexBufferView;

    UINT64 nextFenceValue;
    UINT64 fenceValues[maxFramesInFlight];
    ComPtr<ID3D12Fence> fence;
    HANDLE fenceEvent;
};
static DxState dx{};

const char kShader[] = R"shader(
#define BROT_LOOP_COUNT 200
cbuffer ShaderData : register(b0)
{
  float3x3 rectXform;
  uint metaLoopCount;
  float4 _pad[13]; // (64-12)/4
};

struct PSInput
{
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD;
};

PSInput VSMain(float4 pos : POSITION, float4 uv : TEXCOORD)
{
    PSInput result;
    float3 p = float3(pos.xy, 1.0);
    p = mul(p, rectXform);
    result.pos = float4(p.xy, 0.0, 1.0);
    result.uv = uv.xy;
    return result;
}

float mandelbrot(float2 c) {
  float2 z = float2(0.0, 0.0);
  for (int i = 0; i < BROT_LOOP_COUNT; i++) {
    z = float2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
    if (z.x*z.x + z.y*z.y > 4) break;
  }
  float grade = 0.0;
  if (i < BROT_LOOP_COUNT) {
    grade = float(i) / float(BROT_LOOP_COUNT);
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

float4 PSMain(PSInput input) : SV_TARGET
{
  if (input.uv.x < 0.25 || input.uv.x > 0.75) {
    return float4(1.0, 0.0, 0.0, 1.0);
  }

  float grade = 0.0;
  uint seed = (uint)input.pos.x;
  for (uint i = 0; i < metaLoopCount; i++) {
    float2 p = input.uv + rand2(seed) / float2(1920, 1080);
    float2 c = float2(-2.5 + p.x * 3.5, -1.0 + p.y * 2.0);
    grade += mandelbrot(c);
  }
  grade = grade / float(metaLoopCount);
  return float4(0.0, grade, grade, 1.0);
}
)shader";

struct Window
{
    HWND handle = NULL;
    int width = 0;
    int height = 0;

    ComPtr<ID3D12GraphicsCommandList> commandList;
    ComPtr<IDXGISwapChain3> swapchain;
    HANDLE latencyWaitable;
    ComPtr<ID3D12Resource> renderTargets[numBackbuffers];
    ComPtr<ID3D12CommandAllocator> commandAllocators[maxFramesInFlight];
    ComPtr<ID3D12Resource> constantBuffers[maxFramesInFlight];
    ShaderData* mappedShaderDatas[maxFramesInFlight];

    bool isFullscreen = false;
    bool resizeBuffers = false;
};

static std::vector<Window> windows;

static bool s_run = true;
static bool s_advance = true;
static bool s_advanceOnce = false;
static bool s_drainPresentQueue = false;
static int s_frameCount = 0;

static RECT GetDesktopRect(IDXGISwapChain* swapchain, RECT windowRect)
{
  ComPtr<IDXGIOutput> output;
  DXGI_OUTPUT_DESC outputDesc;

  HRESULT hr = swapchain->GetContainingOutput(&output);
  if (hr == S_OK) {
    assert_hr(output->GetDesc(&outputDesc));
    return outputDesc.DesktopCoordinates;
  }

  // GetContainingOutput may fail, in which case we have to manually iterate all adapter>output pairs and
  // search for the best intersection with the window rectangle.
  ComPtr<IDXGIFactory4> dxgiFactory;
  assert_hr(CreateDXGIFactory2(0, IID_PPV_ARGS(&dxgiFactory)));
  ComPtr<IDXGIAdapter1> adapter;
  for (UINT adapterIdx = 0; SUCCEEDED(dxgiFactory->EnumAdapters1(adapterIdx, &adapter)); adapterIdx++) {
    DXGI_ADAPTER_DESC1 adapterDesc;
    adapter->GetDesc1(&adapterDesc);

    for (UINT outputIdx = 0; SUCCEEDED(adapter->EnumOutputs(outputIdx, &output)); outputIdx++) {
      assert_hr(output->GetDesc(&outputDesc));

      if (outputDesc.DesktopCoordinates.left <= windowRect.left &&
          windowRect.left < outputDesc.DesktopCoordinates.right &&
          outputDesc.DesktopCoordinates.top <= windowRect.top &&
          windowRect.top < outputDesc.DesktopCoordinates.bottom) {
        return outputDesc.DesktopCoordinates;
      }
    }
  }
  assert(!"Could not find desktop rect for window");
  return { 0, 0, 1920, 1080 };
}

static void OnResize(size_t winIdx);

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
        for (size_t i = 0; i < windows.size(); i++) {
          Window& window = windows[i];
          if (window.handle != handle) continue;
          window.width = LOWORD(lparam);
          window.height = HIWORD(lparam);
          OnResize(i);
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

      if ((wparam == 'D') && pressed) {
        s_drainPresentQueue = true;
      }

      if ((wparam == 'F') && pressed) {
        for (size_t i = 0; i < windows.size(); i++) {
          Window& window = windows[i];
          if (!window.swapchain) continue;

          RECT windowRect;
          ::GetWindowRect(window.handle, &windowRect);
          RECT desktopRect = GetDesktopRect(window.swapchain.get(), windowRect);

          if (window.isFullscreen) {
            //            ::SetWindowLongPtr(window.handle, GWL_STYLE, WS_OVERLAPPEDWINDOW);
            ::SetWindowPos(window.handle,
                           HWND_NOTOPMOST,
                           desktopRect.left + 100, desktopRect.top + 100,
                           1920,
                           1080,
                           SWP_FRAMECHANGED | SWP_NOACTIVATE);
            ::ShowWindow(window.handle, SW_NORMAL);
          }
          else {
            //            ::SetWindowLongPtr(window.handle, GWL_STYLE, WS_VISIBLE|WS_POPUP);
            ::SetWindowPos(window.handle,
                           HWND_TOPMOST,
                           desktopRect.left,
                           desktopRect.top,
                           desktopRect.right - desktopRect.left,
                           desktopRect.bottom - desktopRect.top,
                           SWP_FRAMECHANGED | SWP_NOACTIVATE); // FRAMECHANGED needed to apply window style set above.
            ::ShowWindow(window.handle, SW_SHOW);
          }

          window.isFullscreen = !window.isFullscreen;
          window.resizeBuffers = true;
        }
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
  DWORD style = WS_VISIBLE|WS_POPUP;
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
  ::CloseHandle(window.latencyWaitable);
  ::DestroyWindow(window.handle);
}

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

  const char* className = RegisterWindowClass();
  if (!className) return EXIT_FAILURE;

  for (int i = 0; i < monitorRects.size(); i++) {
    bool success = createWindow(className, i, monitorRects[i]);
    if (!success) {
      Log("Failed to create window %d!", i);
      return EXIT_FAILURE;
    }
  }

  {
    UINT dxgiFactoryFlags = 0;
#if defined(FN_DEBUG)
    {
      ComPtr<ID3D12Debug> debugController;
      if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)))) {
        debugController->EnableDebugLayer();
        dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
      }

      ComPtr<ID3D12Debug5> debug5;
      assert_hr(debugController->QueryInterface(IID_PPV_ARGS(&debug5)));
      debug5->SetEnableAutoName(true);
    }
#endif
    ComPtr<IDXGIFactory4> dxgiFactory;
    assert_hr(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&dxgiFactory)));
    ComPtr<IDXGIAdapter1> adapter;
    {
      for (UINT adapterIdx = 0; SUCCEEDED(dxgiFactory->EnumAdapters1(adapterIdx, &adapter)); adapterIdx++) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);

        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
          continue;
        }

        if (SUCCEEDED(D3D12CreateDevice(adapter.get(), D3D_FEATURE_LEVEL_12_0, _uuidof(ID3D12Device), nullptr))) {
          break;
        }
      }
    }
    assert_hr(D3D12CreateDevice(adapter.get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dx.device)));

    D3D12_COMMAND_QUEUE_DESC queueDesc{};
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

    assert_hr(dx.device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&dx.commandQueue)));
    dx.commandQueue->SetName(L"commandQueue");

    {
      D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc{};
      rtvHeapDesc.NumDescriptors = static_cast<UINT>(windows.size() * numBackbuffers);
      rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
      rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
      assert_hr(dx.device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&dx.rtvHeap)));
      dx.rtvHeap->SetName(L"rtvHeap");

      dx.rtvDescriptorSize = dx.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

      D3D12_DESCRIPTOR_HEAP_DESC cbvHeapDesc{};
      cbvHeapDesc.NumDescriptors = static_cast<UINT>(windows.size() * maxFramesInFlight);
      cbvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
      cbvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
      assert_hr(dx.device->CreateDescriptorHeap(&cbvHeapDesc, IID_PPV_ARGS(&dx.cbvHeap)));
      dx.cbvHeap->SetName(L"cbvHeap");

      dx.cbvDescriptorSize = dx.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }

    {
      CD3DX12_DESCRIPTOR_RANGE1 ranges[1];
      CD3DX12_ROOT_PARAMETER1 rootParameters[1];
      ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC);
      rootParameters[0].InitAsDescriptorTable(1, &ranges[0], D3D12_SHADER_VISIBILITY_ALL);

      D3D12_ROOT_SIGNATURE_FLAGS rootSignatureFlags =
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
        D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
        D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
        D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS;

      CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSignatureDesc;
      rootSignatureDesc.Init_1_1(static_cast<UINT>(ArrayCount(rootParameters)), rootParameters, 0, nullptr, rootSignatureFlags);

      ComPtr<ID3DBlob> signature;
      ComPtr<ID3DBlob> error;
      assert_hr(D3DX12SerializeVersionedRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1_1, &signature, &error));
      assert_hr(dx.device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&dx.rootSignature)));
    }

    {
      ComPtr<ID3DBlob> vertexShader;
      ComPtr<ID3DBlob> pixelShader;
#if defined(FN_DEBUG)
      // Enable better shader debugging with the graphics debugging tools.
      UINT compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
      UINT compileFlags = 0;
#endif
      auto CompileShader = [&](const char* entry, const char* target, ID3DBlob** out) {
          ComPtr<ID3DBlob> errors;
          D3DCompile(kShader, ArrayCount(kShader), "shader.hlsl", nullptr, nullptr, entry, target, compileFlags, 0, out, &errors);

          if (errors) {
            Log("Shader compiler error: %s", reinterpret_cast<char*>(errors->GetBufferPointer()));
            assert(false);
          }
      };
      CompileShader("VSMain", "vs_5_0", &vertexShader);
      CompileShader("PSMain", "ps_5_0", &pixelShader);

      D3D12_INPUT_ELEMENT_DESC inputElementDescs[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 8, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
      };

      D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc{};
      psoDesc.InputLayout = { inputElementDescs, static_cast<UINT>(ArrayCount(inputElementDescs)) };
      psoDesc.pRootSignature = dx.rootSignature.get();
      psoDesc.VS = CD3DX12_SHADER_BYTECODE(vertexShader.get());
      psoDesc.PS = CD3DX12_SHADER_BYTECODE(pixelShader.get());
      psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
      psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
      psoDesc.DepthStencilState.DepthEnable = false;
      psoDesc.DepthStencilState.StencilEnable = false;
      psoDesc.SampleMask = UINT_MAX;
      psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
      psoDesc.NumRenderTargets = 1;
      psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
      psoDesc.SampleDesc.Count = 1;
      assert_hr(dx.device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&dx.pipelineState)));
    }

    CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(dx.rtvHeap->GetCPUDescriptorHandleForHeapStart());
    CD3DX12_CPU_DESCRIPTOR_HANDLE cbvHandle(dx.cbvHeap->GetCPUDescriptorHandleForHeapStart());

    for (size_t winIdx = 0; winIdx < windows.size(); winIdx++) {
      Window& window = windows[winIdx];

      DXGI_SWAP_CHAIN_DESC1 swapchainDesc{};
      swapchainDesc.BufferCount = numBackbuffers;
      swapchainDesc.Width = window.width;
      swapchainDesc.Height = window.height;
      swapchainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
      swapchainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
      swapchainDesc.SampleDesc.Count = 1;
      swapchainDesc.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT | DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;

      ComPtr<IDXGISwapChain1> swapchain;
      assert_hr(dxgiFactory->CreateSwapChainForHwnd(
        dx.commandQueue.get(),
        window.handle,
        &swapchainDesc,
        nullptr,
        nullptr,
        &swapchain));

      assert_hr(dxgiFactory->MakeWindowAssociation(window.handle, DXGI_MWA_NO_ALT_ENTER));
      assert_hr(swapchain->QueryInterface(IID_PPV_ARGS(&window.swapchain)));

      window.swapchain->SetMaximumFrameLatency(1);
      window.latencyWaitable = window.swapchain->GetFrameLatencyWaitableObject();
      assert(window.latencyWaitable);

      wchar_t name[32];
      for (uint32_t i = 0; i < numBackbuffers; i++) {
        assert_hr(window.swapchain->GetBuffer(i, IID_PPV_ARGS(&window.renderTargets[i])));
        dx.device->CreateRenderTargetView(window.renderTargets[i].get(), nullptr, rtvHandle);
        swprintf(name, ArrayCount(name), L"window %zu backbuffer %u", winIdx, i);

        window.renderTargets[i]->SetName(name);
        rtvHandle.Offset(1, dx.rtvDescriptorSize);
      }

      const UINT cbSize = sizeof(ShaderData);
      CD3DX12_HEAP_PROPERTIES cbHeapProps(D3D12_HEAP_TYPE_UPLOAD);
      auto cbResourceDesc = CD3DX12_RESOURCE_DESC::Buffer(cbSize);

      for (UINT i = 0; i < maxFramesInFlight; i++) {
        assert_hr(dx.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&window.commandAllocators[i])));
        swprintf(name, ArrayCount(name), L"window %zu command allocator", winIdx);

        {
          assert_hr(dx.device->CreateCommittedResource(
            &cbHeapProps,
            D3D12_HEAP_FLAG_NONE,
            &cbResourceDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&window.constantBuffers[i])
          ));
          swprintf(name, ArrayCount(name), L"window %zu constant buffer %u", winIdx, i);
          window.constantBuffers[i]->SetName(name);

          D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc{};
          cbvDesc.BufferLocation = window.constantBuffers[i]->GetGPUVirtualAddress();
          cbvDesc.SizeInBytes = cbSize;
          dx.device->CreateConstantBufferView(&cbvDesc, cbvHandle);
          cbvHandle.Offset(1, dx.cbvDescriptorSize);

          CD3DX12_RANGE readRange(0, 0);
          // We keep this mapped for the duration of the program.
          assert_hr(window.constantBuffers[i]->Map(0, &readRange, reinterpret_cast<void**>(&window.mappedShaderDatas[i])));
        }
      }

      assert_hr(dx.device->CreateCommandList(
        0, D3D12_COMMAND_LIST_TYPE_DIRECT,
        // The choice of allocator is arbitrary, we just need one for creation. It'll get reset in the render loop.
        window.commandAllocators[0].get(), dx.pipelineState.get(), IID_PPV_ARGS(&window.commandList)));
      swprintf(name, ArrayCount(name), L"window %zu command list", winIdx);
      window.commandList->SetName(name);

      // Command lists get created open, but we want it closed for the main loop, so close now.
      assert_hr(window.commandList->Close());
    }

    {
      DirectX::XMFLOAT2 corners[] = {
        { 0.0f, 0.0f },
        { 1.0f, 0.0f },
        { 0.0f, 1.0f },
        { 1.0f, 1.0f },
      };

      std::vector<Vertex> verts;
      verts.reserve(6);
      for (size_t idx : { 0, 2, 1, 1, 2, 3 }) {
        DirectX::XMFLOAT2 p = corners[idx];
        p.x = p.x * 2.0f - 1.0f;
        p.y = p.y * 2.0f - 1.0f;
        verts.push_back(Vertex { p, corners[idx] });
      }
      size_t bufferSize = verts.size() * sizeof(Vertex);

      // Using an upload buffer is not very efficient, moving this on the GPU to a default buffer would be better.
      // But our data is small, so we don't care for the moment.
      CD3DX12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE_UPLOAD);
      auto resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(bufferSize);
      assert_hr(dx.device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &resourceDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&dx.vertexBuffer)
      ));
      dx.vertexBuffer->SetName(L"vertex buffer");

      uint8_t* bufferAddr;
      CD3DX12_RANGE readRange(0, 0);
      assert_hr(dx.vertexBuffer->Map(0, &readRange, reinterpret_cast<void**>(&bufferAddr)));
      memcpy(bufferAddr, verts.data(), bufferSize);
      dx.vertexBuffer->Unmap(0, nullptr);

      dx.vertexBufferView.BufferLocation = dx.vertexBuffer->GetGPUVirtualAddress();
      dx.vertexBufferView.StrideInBytes = sizeof(Vertex);
      dx.vertexBufferView.SizeInBytes = static_cast<UINT>(bufferSize);
    }

    {
      assert_hr(dx.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&dx.fence)));
      dx.fence->SetName(L"frame fence");
      dx.nextFenceValue = 1;
      dx.fenceEvent = ::CreateEvent(nullptr, false, false, nullptr);
      if (!dx.fenceEvent) {
        assert_hr(HRESULT_FROM_WIN32(::GetLastError()));
      }
    }
  }

  auto PumpMessages = []() {
      ZoneScopedN("PumpMessages");
      for (int i = 0; i < windows.size(); i++) {
        MSG message;
        while (::PeekMessage(&message, windows[i].handle, 0, 0, PM_REMOVE)) {
          ::TranslateMessage(&message);
          ::DispatchMessage(&message);
        }
      }
  };

  int linePos = 0;
  auto frameStart = std::chrono::high_resolution_clock::now();
  while (s_run) {
    FrameMark;

    auto frameEnd = frameStart;
    frameStart = std::chrono::high_resolution_clock::now();

    {
      ZoneScopedN("wait on latency waitables");
      std::vector<HANDLE> waitables(windows.size());
      for (size_t i = 0; i < windows.size(); i++) {
        waitables[i] = windows[i].latencyWaitable;
      }

      DWORD waitResult = ::WaitForMultipleObjectsEx((DWORD)waitables.size(), waitables.data(), true, 1000, true);
      if (waitResult == WAIT_FAILED) {
        Log("Waiting on swapchains failed.");
      } else if (waitResult == WAIT_TIMEOUT) {
        Log("Swapchain wait timed out.");
      } else {
        assert(WAIT_OBJECT_0 <= waitResult && waitResult < WAIT_OBJECT_0 + waitables.size());
      }
    }

    const UINT resourceIdx = s_frameCount % maxFramesInFlight;

    {
      ZoneScopedN("wait for frame fence");
      // Wait to ensure that the resources we're about to use are unused.
      if (dx.fence->GetCompletedValue() < dx.fenceValues[resourceIdx]) {
        assert_hr(dx.fence->SetEventOnCompletion(dx.fenceValues[resourceIdx], dx.fenceEvent));
        WaitForSingleObjectEx(dx.fenceEvent, INFINITE, false);
      }

      // We advance our fence value right before we signal it. This should help to avoid writing code that waits on a
      // value that hasn't been signaled.
    }

    for (uint32_t winIdx = 0; winIdx < windows.size(); winIdx++) {
      if (windows[winIdx].resizeBuffers) {
        OnResize(winIdx);
      }
    }

    for (uint32_t winIdx = 0; winIdx < windows.size(); winIdx++) {

      Window& window = windows[winIdx];
      {
        float rect[4] = {
          (float) window.width / 2.0f, (float) (window.height - linePos % window.height),
          (float) window.width, 100
        };

        float matrix[9] = {
          rect[2]/(float)window.width, 0.0f,                               2.0f*rect[0]/(float)window.width  - 1.0f,
          0.0f,                              rect[3]/(float)window.height, 2.0f*rect[1]/(float)window.height - 1.0f,
          0.0f,                              0.0f,                         1.0f,
        };

        ShaderData shaderData{};
        memcpy(shaderData.rectXform_r1, &matrix[0], sizeof(float) * 3);
        memcpy(shaderData.rectXform_r2, &matrix[3], sizeof(float) * 3);
        memcpy(shaderData.rectXform_r3, &matrix[6], sizeof(float) * 3);
        shaderData.metaLoopCount = 600;

        memcpy(window.mappedShaderDatas[resourceIdx], &shaderData, sizeof(shaderData));
      }

      {
        ZoneScopedN("Record commands");

        assert_hr(window.commandAllocators[resourceIdx]->Reset());
        assert_hr(window.commandList->Reset(window.commandAllocators[resourceIdx].get(), dx.pipelineState.get()));
        window.commandList->SetGraphicsRootSignature(dx.rootSignature.get());

        ID3D12DescriptorHeap* heaps[] = { dx.cbvHeap.get() };
        window.commandList->SetDescriptorHeaps(static_cast<UINT>(ArrayCount(heaps)), heaps);

        CD3DX12_GPU_DESCRIPTOR_HANDLE cbvHandle(dx.cbvHeap->GetGPUDescriptorHandleForHeapStart(),
                                                winIdx * maxFramesInFlight + resourceIdx, dx.cbvDescriptorSize);
        window.commandList->SetGraphicsRootDescriptorTable(0, cbvHandle);

        CD3DX12_VIEWPORT viewport(0.0f, 0.0f, static_cast<float>(window.width), static_cast<float>(window.height));
        CD3DX12_RECT scissorRect(0, 0, window.width, window.height);
        window.commandList->RSSetViewports(1, &viewport);
        window.commandList->RSSetScissorRects(1, &scissorRect);

        UINT backbufferIdx = window.swapchain->GetCurrentBackBufferIndex();
        auto presentToRt = CD3DX12_RESOURCE_BARRIER::Transition(window.renderTargets[backbufferIdx].get(),
                                                                D3D12_RESOURCE_STATE_PRESENT,
                                                                D3D12_RESOURCE_STATE_RENDER_TARGET);
        window.commandList->ResourceBarrier(1, &presentToRt);

        CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(dx.rtvHeap->GetCPUDescriptorHandleForHeapStart(),
                                                winIdx * numBackbuffers + backbufferIdx, dx.rtvDescriptorSize);
        window.commandList->OMSetRenderTargets(1, &rtvHandle, false, nullptr);

        const float clearColor[] = { 0.0f, 0.2f, 0.4f, 1.0f };
        window.commandList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
        window.commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        window.commandList->IASetVertexBuffers(0, 1, &dx.vertexBufferView);
        window.commandList->DrawInstanced(6, 1, 0, 0);

        auto rtToPresent = CD3DX12_RESOURCE_BARRIER::Transition(window.renderTargets[backbufferIdx].get(),
                                                                D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                                D3D12_RESOURCE_STATE_PRESENT);
        window.commandList->ResourceBarrier(1, &rtToPresent);

        assert_hr(window.commandList->Close());
      }
    }

    {
      ZoneScopedN("Execute command lists");
      std::vector<ID3D12CommandList*> commandLists;
      commandLists.reserve(windows.size());
      for (size_t i = 0; i < windows.size(); i++) {
        commandLists.push_back(windows[i].commandList.get());
      }
      dx.commandQueue->ExecuteCommandLists(static_cast<UINT>(commandLists.size()), commandLists.data());
    }

    {
      for (size_t i = 0; i < windows.size(); i++) {
        Window& window = windows[i];

        ZoneScopedN("Present");
        ZoneTextF("Window %zu", i);
        assert_hr(window.swapchain->Present(1, 0));
      }
    }

    // Queue a frame completion signal
    dx.fenceValues[resourceIdx] = dx.nextFenceValue;
    dx.nextFenceValue++;
    assert_hr(dx.commandQueue->Signal(dx.fence.get(), dx.fenceValues[resourceIdx]));

    PumpMessages();

    if (s_advance || s_advanceOnce) {
      linePos++;
    }
    s_advanceOnce = false;

    s_frameCount++;

    {
      constexpr uint32_t numQueueDrainFrames = 6;
      if (s_drainPresentQueue) {
        for (int i = 0; i < numQueueDrainFrames; i++) {
          ZoneScopedN("non-render frame");
          Spinloop(40);
          PumpMessages();
        }
        s_drainPresentQueue = false;
      }
    }
  }

  // Wait for the last submitted frame to finish rendering before freeing resources.
  UINT resourceIdx = (s_frameCount - 1) % maxFramesInFlight; // Account for s_frameCount being incremented at the end of the main loop.
  if (dx.fence->GetCompletedValue() < dx.fenceValues[resourceIdx]) {
    assert_hr(dx.fence->SetEventOnCompletion(dx.fenceValues[resourceIdx], dx.fenceEvent));
    WaitForSingleObjectEx(dx.fenceEvent, INFINITE, false);
  }

  // Make sure all swapchains are out of fullscreen before cleaning them up, since destroying a swapchain in fullscreen
  // mode is an error.
  for (size_t i = 0; i < windows.size(); i++) {
    Window& window = windows[i];
    BOOL fullscreen;
    window.swapchain->GetFullscreenState(&fullscreen, nullptr);
    if (fullscreen) {
      window.swapchain->SetFullscreenState(false, nullptr);
    }
  }

  for (Window& window : windows) {
    destroyWindow(window);
  }

  return EXIT_SUCCESS;
}

static void OnResize(size_t winIdx)
{
  Window& window = windows[winIdx];

  if (!window.swapchain) return;

  uint64_t lastFence = 0;
  for (size_t i = 0; i < maxFramesInFlight; i++) {
    lastFence = std::max(lastFence, dx.fenceValues[i]);
  }
  if (dx.fence->GetCompletedValue() < lastFence) {
    assert_hr(dx.fence->SetEventOnCompletion(lastFence, dx.fenceEvent));
    WaitForSingleObjectEx(dx.fenceEvent, INFINITE, false);
  }

  // Release resources held by the swapchain before resizing
  for (size_t i = 0; i < ArrayCount(window.renderTargets); i++) {
    window.renderTargets[i] = nullptr; // .release() doesn't actually decref and thus release the pointer
  }

  DXGI_SWAP_CHAIN_DESC desc{};
  window.swapchain->GetDesc(&desc);
  assert_hr(window.swapchain->ResizeBuffers(desc.BufferCount, window.width, window.height, desc.BufferDesc.Format, desc.Flags));

  CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(dx.rtvHeap->GetCPUDescriptorHandleForHeapStart());
  rtvHandle.Offset(static_cast<INT>(winIdx * numBackbuffers), dx.rtvDescriptorSize);
  for (uint32_t i = 0; i < numBackbuffers; i++) {
    assert_hr(window.swapchain->GetBuffer(i, IID_PPV_ARGS(&window.renderTargets[i])));
    dx.device->CreateRenderTargetView(window.renderTargets[i].get(), nullptr, rtvHandle);
    rtvHandle.Offset(1, dx.rtvDescriptorSize);
  }

  window.resizeBuffers = false;
}
