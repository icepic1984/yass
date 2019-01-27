#include <chrono>
#include <random>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan.hpp>
#include <glfw.hpp>
#include <GLFW/glfw3.h>
#include <fstream>
using namespace std::chrono_literals;

template <typename T> class Timer {
public:
    Timer() : m_start(std::chrono::system_clock::now())
    {
    }

    T elapsed()
    {
        return std::chrono::duration_cast<T>(std::chrono::system_clock::now() -
                                             m_start);
    }

    void reset()
    {
        m_start = std::chrono::system_clock::now();
    }

private:
    std::chrono::time_point<std::chrono::system_clock> m_start;
};

struct Rgba {
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
};

class ImageView {
public:
    ImageView(int width, int height, int depth, void* memory)
        : m_width(width), m_height(height), m_depth(depth),
          m_memory(static_cast<uint8_t*>(memory))
    {
    }

    uint8_t& operator()(int x, int y, int d)
    {
        return m_memory[y * m_width * m_depth + (x * m_depth + d)];
    }

    uint8_t operator()(int x, int y, int d) const
    {
        return 0;
    }

    int width()
    {
        return m_width;
    }

    int height()
    {
        return m_height;
    }

    int depth()
    {
        return m_depth;
    }

private:
    int m_width;
    int m_height;
    int m_depth;

    uint8_t* m_memory;
};

// #define GLM_FORCE_RADIANS
// #define GLM_FORCE_DEPTH_ZERO_TO_ONE
// #include <glm/mat4x4.hpp>
// #include <glm/vec4.hpp>

#include <array>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <optional>
#include <limits>
#include <vector>

const bool enableValidationLayer = true;

const int WIDTH = 800;
const int HEIGHT = 600;

// clang-format off
const std::vector<const char*> validationLayers{
    "VK_LAYER_LUNARG_standard_validation",
    "VK_LAYER_LUNARG_parameter_validation",
    "VK_LAYER_LUNARG_object_tracker",
    "VK_LAYER_LUNARG_core_validation"

};
// clang-format on

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

struct SwapChainSupportDetails {
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
};

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData)
{

    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

    return VK_FALSE;
}

SwapChainSupportDetails
querySwapChainSupport(const vk::PhysicalDevice& physicalDevice,
                      const vk::UniqueSurfaceKHR& surface)
{
    SwapChainSupportDetails result;
    result.capabilities =
        physicalDevice.getSurfaceCapabilitiesKHR(surface.get());
    result.formats = physicalDevice.getSurfaceFormatsKHR(surface.get());
    result.presentModes =
        physicalDevice.getSurfacePresentModesKHR(surface.get());

    return result;
}

std::vector<const char*> getRequiredExtensionsForGlfw()
{
    uint32_t extensionCount = 0;
    const char** extensions = nullptr;
    extensions = glfwGetRequiredInstanceExtensions(&extensionCount);
    if (!extensions) {
        throw std::runtime_error(
            "Could not retrieve list of required vk extensions");
    }
    std::vector<const char*> result(extensionCount);
    std::copy(extensions, extensions + extensionCount, result.begin());
    if (enableValidationLayer)
        result.push_back("VK_EXT_debug_utils");
    return result;
}

bool checkValidationLayer()
{
    auto layerProperties = vk::enumerateInstanceLayerProperties();

    for (const auto& validationLayer : validationLayers) {
        auto found = std::find_if(
            layerProperties.begin(), layerProperties.end(),
            [&validationLayer](const vk::LayerProperties& layer) {
                if (std::strcmp(validationLayer, layer.layerName) == 0)
                    return true;
                return false;
            });

        if (found == layerProperties.end())
            return false;
    }
    return true;
}

std::vector<vk::ExtensionProperties> filterExtensions(
    const std::vector<vk::ExtensionProperties>& availableExtensions,
    const std::vector<const char*>& requiredExtensionNames)
{
    std::vector<vk::ExtensionProperties> result;
    std::copy_if(
        availableExtensions.begin(), availableExtensions.end(),
        std::back_inserter(result),
        [&requiredExtensionNames](const vk::ExtensionProperties& extension) {
            for (const auto& iter : requiredExtensionNames)
                if (std::strcmp(iter, extension.extensionName) == 0)
                    return true;
            return false;
        });
    return result;
}

bool meetExtensionRequirements(
    const std::vector<vk::ExtensionProperties>& availableExtensions,
    const std::vector<const char*>& requiredExtensionNames)
{
    auto result = filterExtensions(availableExtensions, requiredExtensionNames);
    return result.size() == requiredExtensionNames.size();
}

bool checkDeviceExtensionSupport(const vk::PhysicalDevice& physicalDevice)
{
    auto extensions = physicalDevice.enumerateDeviceExtensionProperties();
    std::vector<vk::ExtensionProperties> result;

    std::copy_if(extensions.begin(), extensions.end(),
                 std::back_inserter(result),
                 [](const vk::ExtensionProperties& property) {
                     for (const auto& iter : deviceExtensions) {
                         if (strcmp(property.extensionName, iter) == 0)
                             return true;
                     }
                     return false;
                 });
    return result.size() == deviceExtensions.size();
}

bool isDeviceSuitable(const vk::PhysicalDevice& device)
{
    auto properties = device.getProperties();
    auto features = device.getFeatures();
    // Test if device is a dedicated graphic card and supports a
    // geometry shader.
    return properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu &&
           features.geometryShader && checkDeviceExtensionSupport(device);
}

vk::PhysicalDevice pickDevice(const vk::UniqueInstance& instance)
{
    // Pick first device which meets the requirements (aka is a
    // dedicated graphic card and has a geometry shader)
    auto devices = instance->enumeratePhysicalDevices();
    for (const auto& iter : instance->enumeratePhysicalDevices()) {
        if (isDeviceSuitable(iter))
            return iter;
    }
    throw std::runtime_error(
        "No graphic cards with vulkan support found on system.");
    return vk::PhysicalDevice();
}

void printQueueProperties(const vk::PhysicalDevice& device)
{
    std::cout << "Queue properties of: " << device.getProperties().deviceName
              << std::endl;

    auto queues = device.getQueueFamilyProperties();
    for (const auto& iter : queues) {
        std::cout << "Properties ("
                  << static_cast<unsigned int>(iter.queueFlags)
                  << "): Queue count = " << iter.queueCount << " Familie = ";
        if (iter.queueFlags & vk::QueueFlagBits::eProtected)
            std::cout << "Protected ";
        if (iter.queueFlags & vk::QueueFlagBits::eGraphics)
            std::cout << "Graphic ";
        if (iter.queueFlags & vk::QueueFlagBits::eTransfer)
            std::cout << "Transfer ";
        if (iter.queueFlags & vk::QueueFlagBits::eCompute)
            std::cout << "Compute ";
        if (iter.queueFlags & vk::QueueFlagBits::eSparseBinding)
            std::cout << "SparseBinding ";
        std::cout << std::endl;
    }
}

void printSurfaceFormats(const vk::SurfaceFormatKHR& format)
{
    switch (format.format) {
    case vk::Format::eUndefined:
        std::cout << "Undefined" << std::endl;
        break;
    case vk::Format::eB8G8R8A8Unorm:
        std::cout << "eB8G8R8A8Unorm" << std::endl;
        break;
    default:
        std::cout << "Unkown" << std::endl;
        break;
    }
}

vk::SurfaceFormatKHR chooseSwapSurfaceFormat(
    const std::vector<vk::SurfaceFormatKHR>& availableFormats)
{
    if (availableFormats.size() == 1 &&
        availableFormats[0].format == vk::Format::eUndefined) {
        return {vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear};
    }

    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == vk::Format::eB8G8R8A8Unorm &&
            availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return availableFormat;
        }
    }

    throw std::runtime_error("No swap format found");
    return availableFormats[0];
}

vk::PresentModeKHR chooseSwapPresentMode(
    const std::vector<vk::PresentModeKHR> availablePresentModes)
{
    vk::PresentModeKHR bestMode = vk::PresentModeKHR::eFifo;

    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
            return availablePresentMode;
        } else if (availablePresentMode == vk::PresentModeKHR::eImmediate) {
            bestMode = availablePresentMode;
        }
    }
    return bestMode;
}

void printSurfaceCapabilities(const vk::SurfaceCapabilitiesKHR& capabilities)
{
    std::cout << "Width: " << capabilities.currentExtent.width << std::endl;
    std::cout << "Height: " << capabilities.currentExtent.height << std::endl;
    std::cout << "Min width : " << capabilities.minImageExtent.width
              << std::endl;
    std::cout << "Min height : " << capabilities.minImageExtent.height
              << std::endl;
    std::cout << "Max width : " << capabilities.maxImageExtent.width
              << std::endl;
    std::cout << "Max height : " << capabilities.maxImageExtent.height
              << std::endl;
    std::cout << "Min Image count: " << capabilities.minImageCount << std::endl;
    std::cout << "Max Image count: " << capabilities.maxImageCount << std::endl;
}

vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities,
                              uint32_t width, uint32_t height)
{
    if (capabilities.currentExtent.width !=
        std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    } else {
        vk::Extent2D actualExtent = {width, height};

        actualExtent.width = std::max(
            capabilities.minImageExtent.width,
            std::min(capabilities.maxImageExtent.width, actualExtent.width));
        actualExtent.height = std::max(
            capabilities.minImageExtent.height,
            std::min(capabilities.maxImageExtent.height, actualExtent.height));

        return actualExtent;
    }
}

vk::UniqueSwapchainKHR
createSwapChain(const vk::UniqueDevice& device,
                const vk::UniqueSurfaceKHR& surface, const vk::Extent2D& extent,
                const vk::PresentModeKHR& presentMode,
                const vk::SurfaceFormatKHR& surfaceFormat)
{
    // For tripple buffering
    const uint32_t imageCount = 3;
    vk::SwapchainCreateInfoKHR info;
    info.setSurface(surface.get());
    info.setMinImageCount(imageCount);
    info.setImageFormat(surfaceFormat.format);
    info.setImageExtent(extent);
    info.setImageColorSpace(surfaceFormat.colorSpace);
    info.setImageArrayLayers(1);
    info.setImageUsage(vk::ImageUsageFlagBits::eColorAttachment |
                       vk::ImageUsageFlagBits::eTransferDst);
    // Graphic and presentation are on same queue therefore choose exclusive
    info.setImageSharingMode(vk::SharingMode::eExclusive);
    info.setPreTransform(vk::SurfaceTransformFlagBitsKHR::eIdentity);
    info.setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque);
    info.setPresentMode(presentMode);
    info.setClipped(true);
    info.setOldSwapchain(nullptr);

    return device->createSwapchainKHRUnique(info);
}

std::optional<uint32_t> findQueueFamilies(const vk::PhysicalDevice& device)
{

    auto queues = device.getQueueFamilyProperties();
    for (std::size_t i = 0; i < queues.size(); ++i) {
        if (queues[i].queueFlags & vk::QueueFlagBits::eGraphics)
            return static_cast<uint32_t>(i);
    }
    return std::nullopt;
}

vk::UniqueInstance
createInstance(const std::vector<const char*>& requiredExtensions)
{
    vk::ApplicationInfo appInfo{"Test"};

    // Setup InstanceCreateInfo struct which is used to configure
    // the vulkan instance.
    vk::InstanceCreateInfo createInfo{
        vk::InstanceCreateFlags(),
        &appInfo,
        0,
        nullptr,
        static_cast<uint32_t>(requiredExtensions.size()),
        requiredExtensions.data()};

    if (checkValidationLayer() && enableValidationLayer) {
        createInfo.setEnabledLayerCount(
            static_cast<uint32_t>(validationLayers.size()));
        createInfo.setPpEnabledLayerNames(validationLayers.data());
    }

    // Create vulkan instance
    return vk::createInstanceUnique(createInfo);
}

vk::UniqueDevice createDevice(const vk::PhysicalDevice& physicalDevice,
                              uint32_t queueFamilyIndex)
{
    // Create device from physical device with one queue.
    float priority = 1.0f;
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo(
        vk::DeviceQueueCreateFlags(), queueFamilyIndex, 1, &priority);
    vk::DeviceCreateInfo deviceCreateInfo(vk::DeviceCreateFlags(), 1,
                                          &deviceQueueCreateInfo);
    if (checkValidationLayer() && enableValidationLayer) {
        deviceCreateInfo.setEnabledLayerCount(
            static_cast<uint32_t>(validationLayers.size()));
        deviceCreateInfo.setPpEnabledLayerNames(validationLayers.data());
    }
    deviceCreateInfo.setEnabledExtensionCount(
        static_cast<uint32_t>(deviceExtensions.size()));
    deviceCreateInfo.setPpEnabledExtensionNames(deviceExtensions.data());
    return physicalDevice.createDeviceUnique(deviceCreateInfo);
}

auto createDebugMessenger(const vk::UniqueInstance& instance,
                          const vk::DispatchLoaderDynamic& dldi)
{
    vk::DebugUtilsMessengerCreateInfoEXT debugInfo(
        vk::DebugUtilsMessengerCreateFlagsEXT(),
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
        vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral,
        &debugCallback, nullptr);

    // Initialize dynamic dispatch
    // Register debug callback for layer validation
    return instance->createDebugUtilsMessengerEXTUnique(debugInfo, nullptr,
                                                        dldi);
}

std::vector<vk::UniqueImageView>
createImageViewsFromSwapChain(const vk::UniqueDevice& device,
                              const std::vector<vk::Image>& swapChainImages,
                              const vk::SurfaceFormatKHR& format)
{
    std::vector<vk::UniqueImageView> views;
    for (const auto& iter : swapChainImages) {
        vk::ImageViewCreateInfo info;
        info.setImage(iter);
        info.setViewType(vk::ImageViewType::e2D);
        info.setFormat(format.format);
        vk::ComponentMapping components(vk::ComponentSwizzle::eIdentity,
                                        vk::ComponentSwizzle::eIdentity,
                                        vk::ComponentSwizzle::eIdentity);
        info.setComponents(components);
        vk::ImageSubresourceRange range;
        range.setAspectMask(vk::ImageAspectFlagBits::eColor);
        range.setBaseMipLevel(0);
        range.setLevelCount(1);
        range.setBaseArrayLayer(0);
        range.setLayerCount(1);
        info.setSubresourceRange(range);
        views.push_back(device->createImageViewUnique(info));
    }
    return views;
}

std::vector<char> readShader(const std::string& str)
{
    std::ifstream file(str, std::ios::ate | std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("File not found");
    auto pos = file.tellg();
    std::vector<char> result(pos);
    file.seekg(0, std::ios::beg);
    file.read(result.data(), pos);
    return result;
}

vk::UniqueShaderModule createShaderModule(const vk::UniqueDevice& device,
                                          const std::vector<char>& code)
{
    vk::ShaderModuleCreateInfo info;
    info.setCodeSize(code.size());
    info.setPCode(reinterpret_cast<const uint32_t*>(code.data()));
    return device->createShaderModuleUnique(info);
}

vk::DescriptorSetLayout
createDescriptorSetLayout(const vk::UniqueDevice& device)
{
    vk::DescriptorSetLayoutBinding binding;
    binding.setBinding(0);
    binding.setDescriptorType(vk::DescriptorType::eUniformBuffer);
    binding.setStageFlags(vk::ShaderStageFlagBits::eVertex);
    binding.setDescriptorCount(1);
    binding.setPImmutableSamplers(nullptr);

    vk::DescriptorSetLayoutCreateInfo descriptorSetCreateInfo;
    descriptorSetCreateInfo.setBindingCount(1);
    descriptorSetCreateInfo.setPBindings(&binding);

    return device->createDescriptorSetLayout(descriptorSetCreateInfo);
}

std::vector<vk::DescriptorSetLayout>
createDescriptorSetLayouts(const vk::UniqueDevice& device, std::size_t number)
{
    std::vector<vk::DescriptorSetLayout> layouts;
    for (int i = 0; i < number; ++i)
        layouts.push_back(createDescriptorSetLayout(device));
    return layouts;
}

std::tuple<vk::UniqueRenderPass, vk::UniquePipeline, vk::UniquePipelineLayout>
createPipeline(const vk::UniqueDevice& device,
               std::vector<vk::DescriptorSetLayout>& descriptorSetLayouts,
               const vk::Extent2D& extent,
               const vk::SurfaceFormatKHR& surfaceFormat)
{

    auto vertShader = createShaderModule(device, readShader("vert.spv"));
    auto fragShader = createShaderModule(device, readShader("frag.spv"));

    vk::PipelineShaderStageCreateInfo vertStageInfo;
    vertStageInfo.setStage(vk::ShaderStageFlagBits::eVertex);
    vertStageInfo.setPName("main");
    vertStageInfo.setModule(vertShader.get());

    vk::PipelineShaderStageCreateInfo fragStageInfo;
    fragStageInfo.setStage(vk::ShaderStageFlagBits::eFragment);
    fragStageInfo.setPName("main");
    fragStageInfo.setModule(fragShader.get());

    std::vector<vk::PipelineShaderStageCreateInfo> stages;
    stages.push_back(vertStageInfo);
    stages.push_back(fragStageInfo);

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo;
    vertexInputInfo.setVertexBindingDescriptionCount(0);
    // vertexInputInfo.setPVertexBindingDescriptions(nullptr);
    vertexInputInfo.setVertexAttributeDescriptionCount(0);
    // vertexInputInfo.setPVertexAttributeDescriptions(nullptr);

    vk::PipelineInputAssemblyStateCreateInfo inputAssemblyInfo;
    inputAssemblyInfo.setTopology(vk::PrimitiveTopology::eTriangleList);
    inputAssemblyInfo.setPrimitiveRestartEnable(false);

    vk::Viewport viewport;
    viewport.setX(0.0f);
    viewport.setY(0.0f);
    viewport.setWidth(static_cast<float>(extent.width));
    viewport.setHeight(static_cast<float>(extent.height));
    viewport.setMinDepth(0.0f);
    viewport.setMaxDepth(1.0f);

    vk::Rect2D scissor;
    scissor.setOffset({0, 0});
    scissor.setExtent(extent);

    vk::PipelineViewportStateCreateInfo viewPortStateInfo;
    viewPortStateInfo.setViewportCount(1);
    viewPortStateInfo.setPViewports(&viewport);
    viewPortStateInfo.setScissorCount(1);
    viewPortStateInfo.setPScissors(&scissor);

    vk::PipelineRasterizationStateCreateInfo rasterizerInfo;
    rasterizerInfo.setRasterizerDiscardEnable(false);
    rasterizerInfo.setDepthClampEnable(false);
    rasterizerInfo.setPolygonMode(vk::PolygonMode::eFill);
    rasterizerInfo.setCullMode(vk::CullModeFlagBits::eBack);
    rasterizerInfo.setFrontFace(vk::FrontFace::eClockwise);
    rasterizerInfo.setLineWidth(1.0f);

    vk::PipelineMultisampleStateCreateInfo multiSampleInfo;
    multiSampleInfo.setSampleShadingEnable(false);
    multiSampleInfo.setRasterizationSamples(vk::SampleCountFlagBits::e1);

    vk::PipelineColorBlendAttachmentState colorBlendAttachment;
    colorBlendAttachment.setColorWriteMask(
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
    colorBlendAttachment.setBlendEnable(false);

    vk::PipelineColorBlendStateCreateInfo colorBlendInfo;
    colorBlendInfo.setLogicOpEnable(false);
    colorBlendInfo.setLogicOp(vk::LogicOp::eCopy);
    colorBlendInfo.setAttachmentCount(1);
    colorBlendInfo.setPAttachments(&colorBlendAttachment);
    colorBlendInfo.setBlendConstants({0.0f, 0.0f, 0.0f, 0.0f});

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    pipelineLayoutInfo.setSetLayoutCount(
        static_cast<uint32_t>(descriptorSetLayouts.size()));
    pipelineLayoutInfo.setPSetLayouts(descriptorSetLayouts.data());

    auto pipelineLayout =
        device->createPipelineLayoutUnique(pipelineLayoutInfo);

    vk::AttachmentDescription colorAttachment;
    colorAttachment.setFormat(surfaceFormat.format);
    colorAttachment.setSamples(vk::SampleCountFlagBits::e1);
    colorAttachment.setLoadOp(vk::AttachmentLoadOp::eClear);
    colorAttachment.setStoreOp(vk::AttachmentStoreOp::eStore);
    colorAttachment.setStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
    colorAttachment.setStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
    colorAttachment.setInitialLayout(vk::ImageLayout::eUndefined);
    colorAttachment.setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

    vk::AttachmentReference colorAttachmentRef;
    colorAttachmentRef.setAttachment(0);
    colorAttachmentRef.setLayout(vk::ImageLayout::eColorAttachmentOptimal);

    vk::SubpassDescription subpass;
    subpass.setPipelineBindPoint(vk::PipelineBindPoint::eGraphics);
    subpass.setColorAttachmentCount(1);
    subpass.setPColorAttachments(&colorAttachmentRef);

    vk::SubpassDependency dependency;
    dependency.setSrcSubpass(VK_SUBPASS_EXTERNAL);
    dependency.setDstSubpass(0);
    dependency.setSrcStageMask(
        vk::PipelineStageFlagBits::eColorAttachmentOutput);
    dependency.setSrcAccessMask(vk::AccessFlags{});
    dependency.setDstStageMask(
        vk::PipelineStageFlagBits::eColorAttachmentOutput);
    dependency.setDstAccessMask(vk::AccessFlagBits::eColorAttachmentRead |
                                vk::AccessFlagBits::eColorAttachmentWrite);

    vk::RenderPassCreateInfo renderPassInfo;
    renderPassInfo.setAttachmentCount(1);
    renderPassInfo.setPAttachments(&colorAttachment);
    renderPassInfo.setSubpassCount(1);
    renderPassInfo.setPSubpasses(&subpass);
    renderPassInfo.setDependencyCount(1);
    renderPassInfo.setPDependencies(&dependency);

    auto renderPass = device->createRenderPassUnique(renderPassInfo);

    vk::GraphicsPipelineCreateInfo pipelineInfo;
    pipelineInfo.setStageCount(2);
    pipelineInfo.setPStages(stages.data());
    pipelineInfo.setPVertexInputState(&vertexInputInfo);
    pipelineInfo.setPInputAssemblyState(&inputAssemblyInfo);
    pipelineInfo.setPViewportState(&viewPortStateInfo);
    pipelineInfo.setPMultisampleState(&multiSampleInfo);
    pipelineInfo.setPColorBlendState(&colorBlendInfo);
    pipelineInfo.setPRasterizationState(&rasterizerInfo);
    pipelineInfo.setLayout(pipelineLayout.get());
    pipelineInfo.setRenderPass(renderPass.get());
    pipelineInfo.setSubpass(0);
    pipelineInfo.setBasePipelineHandle(vk::Pipeline{});

    auto pipeline =
        device->createGraphicsPipelineUnique(vk::PipelineCache{}, pipelineInfo);
    return std::make_tuple(std::move(renderPass), std::move(pipeline),
                           std::move(pipelineLayout));
}

std::vector<vk::UniqueFramebuffer>
createFrameBuffers(const vk::UniqueDevice& device,
                   const vk::UniqueRenderPass& renderPass,
                   const std::vector<vk::UniqueImageView>& swapChainImageViews,
                   const vk::Extent2D& extent)
{

    std::vector<vk::UniqueFramebuffer> swapChainFramebuffers;

    for (const auto& iter : swapChainImageViews) {
        vk::FramebufferCreateInfo frameBufferInfo;
        frameBufferInfo.setRenderPass(renderPass.get());
        frameBufferInfo.setAttachmentCount(1);
        frameBufferInfo.setPAttachments(&iter.get());
        frameBufferInfo.setWidth(extent.width);
        frameBufferInfo.setHeight(extent.height);
        frameBufferInfo.setLayers(1);
        swapChainFramebuffers.push_back(
            device->createFramebufferUnique(frameBufferInfo));
    }
    return swapChainFramebuffers;
}

vk::UniqueCommandPool createCommandPool(const vk::UniqueDevice& device,
                                        uint32_t index)
{
    vk::CommandPoolCreateInfo commandPoolInfo;
    commandPoolInfo.setQueueFamilyIndex(index);
    return device->createCommandPoolUnique(commandPoolInfo);
}

std::vector<vk::CommandBuffer> createDrawCommand(
    const vk::UniqueDevice& device, const vk::UniqueCommandPool& commandPool,
    const vk::UniquePipeline& pipeline, const vk::UniqueRenderPass& renderPass,
    const std::vector<vk::UniqueFramebuffer>& swapChainFrameBuffers,
    const vk::UniquePipelineLayout& pipelineLayout,
    const std::vector<vk::DescriptorSet>& descriptorSets,
    const vk::Extent2D& extent)
{
    // Allocating command buffers (one for each swap image)
    vk::CommandBufferAllocateInfo allocInfo;
    allocInfo.setCommandPool(commandPool.get());
    allocInfo.setLevel(vk::CommandBufferLevel::ePrimary);
    allocInfo.setCommandBufferCount(
        static_cast<uint32_t>(swapChainFrameBuffers.size()));

    std::vector<vk::CommandBuffer> commandBuffers =
        device->allocateCommandBuffers(allocInfo);

    for (std::size_t i = 0; i < commandBuffers.size(); ++i) {

        // Start recording of command buffer
        vk::CommandBufferBeginInfo beginInfo;
        beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eSimultaneousUse);
        commandBuffers[i].begin(beginInfo);

        vk::RenderPassBeginInfo renderPassBeginInfo;
        renderPassBeginInfo.setRenderPass(renderPass.get());
        renderPassBeginInfo.setFramebuffer(swapChainFrameBuffers[i].get());
        renderPassBeginInfo.setRenderArea(vk::Rect2D({0, 0}, extent));
        vk::ClearValue clearColor(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f});
        renderPassBeginInfo.setClearValueCount(1);
        renderPassBeginInfo.setPClearValues(&clearColor);
        commandBuffers[i].beginRenderPass(renderPassBeginInfo,
                                          vk::SubpassContents::eInline);
        commandBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics,
                                       pipeline.get());
        commandBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                             pipelineLayout.get(), 0, 1,
                                             &descriptorSets[i], 0, nullptr);
        commandBuffers[i].draw(3, 1, 0, 0);
        commandBuffers[i].endRenderPass();
        commandBuffers[i].end();
    }
    return commandBuffers;
}

std::pair<vk::UniqueSemaphore, vk::UniqueSemaphore>
createSemaphores(const vk::UniqueDevice& device)
{
    vk::SemaphoreCreateInfo semaphoreCreateInfo;

    return std::make_pair(device->createSemaphoreUnique(semaphoreCreateInfo),
                          device->createSemaphoreUnique(semaphoreCreateInfo));
}

uint32_t findMemoryType(const vk::PhysicalDevice& physicalDevice,
                        uint32_t typeFilter, vk::MemoryPropertyFlags properties)
{
    auto deviceMemProps = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < deviceMemProps.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            ((deviceMemProps.memoryTypes[i].propertyFlags & properties) ==
             properties)) {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type!");
    return 0;
}

std::pair<vk::UniqueBuffer, vk::UniqueDeviceMemory>
createBuffer(const vk::PhysicalDevice& physicalDevice,
             const vk::UniqueDevice& device, vk::DeviceSize size,
             vk::BufferUsageFlags useage, vk::MemoryPropertyFlags properties)
{
    vk::BufferCreateInfo bufferCreateInfo;
    bufferCreateInfo.setSize(size);
    bufferCreateInfo.setUsage(useage);
    bufferCreateInfo.setSharingMode(vk::SharingMode::eExclusive);

    auto buffer = device->createBufferUnique(bufferCreateInfo);
    vk::MemoryRequirements memoryRequirements =
        device->getBufferMemoryRequirements(buffer.get());
    vk::MemoryAllocateInfo memAllocInfo;
    memAllocInfo.setAllocationSize(memoryRequirements.size);
    memAllocInfo.setMemoryTypeIndex(findMemoryType(
        physicalDevice, memoryRequirements.memoryTypeBits, properties));
    auto memory = device->allocateMemoryUnique(memAllocInfo);

    device->bindBufferMemory(buffer.get(), memory.get(), 0);
    return std::make_pair(std::move(buffer), std::move(memory));
}

std::pair<vk::UniqueImage, vk::UniqueDeviceMemory>
createImage(const vk::PhysicalDevice& physicalDevice,
            const vk::UniqueDevice& device, uint32_t width, uint32_t height,
            vk::Format format, vk::ImageLayout initialLayout,
            vk::ImageTiling tiling, vk::ImageUsageFlags usage,
            vk::MemoryPropertyFlags properties)
{
    vk::ImageCreateInfo imageInfo;
    imageInfo.setImageType(vk::ImageType::e2D);
    imageInfo.setExtent({width, height, 1});
    imageInfo.setMipLevels(1);
    imageInfo.setArrayLayers(1);
    imageInfo.setFormat(format);
    imageInfo.setTiling(tiling);
    imageInfo.setInitialLayout(initialLayout);
    imageInfo.setUsage(usage);
    imageInfo.setSamples(vk::SampleCountFlagBits::e1);
    imageInfo.setSharingMode(vk::SharingMode::eExclusive);
    auto image = device->createImageUnique(imageInfo);

    auto memoryRequirements = device->getImageMemoryRequirements(image.get());

    vk::MemoryAllocateInfo memAllocInfo;
    memAllocInfo.setAllocationSize(memoryRequirements.size);
    memAllocInfo.setMemoryTypeIndex(findMemoryType(
        physicalDevice, memoryRequirements.memoryTypeBits, properties));

    auto memory = device->allocateMemoryUnique(memAllocInfo);
    device->bindImageMemory(image.get(), memory.get(), 0);

    return std::make_pair(std::move(image), std::move(memory));
}
void copyBufferToImage(const vk::UniqueDevice& device, const vk::Queue& queue,
                       const vk::UniqueCommandPool& commandPool,
                       const vk::UniqueBuffer& buffer,
                       const vk::UniqueImage& image, uint32_t width,
                       uint32_t height)
{
    vk::CommandBufferAllocateInfo allocInfo;
    allocInfo.setCommandPool(commandPool.get());
    allocInfo.setLevel(vk::CommandBufferLevel::ePrimary);
    allocInfo.setCommandBufferCount(1);
    auto command = device->allocateCommandBuffersUnique(allocInfo);

    vk::CommandBufferBeginInfo beginInfo;
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    command.front()->begin(beginInfo);

    vk::BufferImageCopy region;
    region.setBufferOffset(0);
    region.setBufferRowLength(0);
    region.setBufferImageHeight(0);
    region.setImageSubresource(
        vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1});
    region.setImageOffset({0, 0, 0});
    region.setImageExtent({width, height, 1});

    command.front()->copyBufferToImage(buffer.get(), image.get(),
                                       vk::ImageLayout::eTransferDstOptimal, 1,
                                       &region);
    command.front()->end();
    vk::SubmitInfo submitInfo;
    submitInfo.setCommandBufferCount(1);
    submitInfo.setPCommandBuffers(&command.front().get());
    queue.submit(1, &submitInfo, vk::Fence{});
    queue.waitIdle();
}

void copyBufferData(const vk::UniqueDevice& device, const vk::Queue& queue,
                    const vk::UniqueCommandPool& commandPool,
                    const vk::UniqueBuffer& src, const vk::UniqueBuffer& dst,
                    const vk::DeviceSize& size)
{
    vk::CommandBufferAllocateInfo allocInfo;
    allocInfo.setCommandPool(commandPool.get());
    allocInfo.setLevel(vk::CommandBufferLevel::ePrimary);
    allocInfo.setCommandBufferCount(1);
    auto command = device->allocateCommandBuffersUnique(allocInfo);

    vk::CommandBufferBeginInfo beginInfo;
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    command.front()->begin(beginInfo);

    vk::BufferCopy bufferCopy;
    bufferCopy.setSize(size);
    bufferCopy.setSrcOffset(0);
    bufferCopy.setDstOffset(0);
    command.front()->copyBuffer(src.get(), dst.get(), 1, &bufferCopy);
    command.front()->end();
    vk::SubmitInfo submitInfo;
    submitInfo.setCommandBufferCount(1);
    submitInfo.setPCommandBuffers(&command.front().get());
    queue.submit(1, &submitInfo, vk::Fence{});
    queue.waitIdle();
}

void transitionImageLayout(const vk::UniqueDevice& device,
                           const vk::Queue& queue,
                           const vk::UniqueCommandPool& commandPool,
                           const vk::Image& image, const vk::Format& format,
                           vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
                           vk::AccessFlags srcAccessMask,
                           vk::AccessFlags dstAccessMask,
                           vk::PipelineStageFlags srcStageMask,
                           vk::PipelineStageFlags dstStageMask)
{
    vk::CommandBufferAllocateInfo allocInfo;
    allocInfo.setCommandPool(commandPool.get());
    allocInfo.setLevel(vk::CommandBufferLevel::ePrimary);
    allocInfo.setCommandBufferCount(1);
    auto command = device->allocateCommandBuffersUnique(allocInfo);

    vk::CommandBufferBeginInfo beginInfo;
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    command.front()->begin(beginInfo);

    vk::ImageMemoryBarrier barrier;
    barrier.setOldLayout(oldLayout);
    barrier.setNewLayout(newLayout);
    barrier.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
    barrier.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
    barrier.setImage(image);
    barrier.setSrcAccessMask(srcAccessMask);
    barrier.setDstAccessMask(dstAccessMask);
    barrier.setSubresourceRange(
        vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});

    command.front()->pipelineBarrier(srcStageMask, dstStageMask,
                                     vk::DependencyFlags{}, 0, nullptr, 0,
                                     nullptr, 1, &barrier);

    command.front()->end();
    vk::SubmitInfo submitInfo;
    submitInfo.setCommandBufferCount(1);
    submitInfo.setPCommandBuffers(&command.front().get());
    queue.submit(1, &submitInfo, vk::Fence{});
    queue.waitIdle();
}

std::pair<vk::UniqueImage, vk::UniqueDeviceMemory>
createTextureImage(const vk::PhysicalDevice& physicalDevice,
                   const vk::UniqueDevice& device, const vk::Queue& queue,
                   const vk::UniqueCommandPool& commandPool)
{
    std::vector<uint32_t> data(WIDTH * HEIGHT * 4, 0xFF0000FF);

    vk::DeviceSize imageSize = WIDTH * HEIGHT * 4;

    auto stagingBuffer =
        createBuffer(physicalDevice, device, imageSize,
                     vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible |
                         vk::MemoryPropertyFlagBits::eHostCoherent);

    void* map = device->mapMemory(stagingBuffer.second.get(), 0, imageSize);
    std::memcpy(map, data.data(), imageSize);
    device->unmapMemory(stagingBuffer.second.get());

    auto imageBuffer = createImage(
        physicalDevice, device, WIDTH, HEIGHT, vk::Format::eR8G8B8A8Unorm,
        vk::ImageLayout::eUndefined, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst |
            vk::ImageUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    transitionImageLayout(
        device, queue, commandPool, imageBuffer.first.get(),
        vk::Format::eR8G8B8A8Unorm, vk::ImageLayout::eUndefined,
        vk::ImageLayout::eTransferDstOptimal, vk::AccessFlags{0},
        vk::AccessFlagBits::eTransferWrite,
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eTransfer);

    copyBufferToImage(device, queue, commandPool, stagingBuffer.first,
                      imageBuffer.first, WIDTH, HEIGHT);
    transitionImageLayout(
        device, queue, commandPool, imageBuffer.first.get(),
        vk::Format::eR8G8B8A8Unorm, vk::ImageLayout::eTransferDstOptimal,
        vk::ImageLayout::eTransferSrcOptimal,
        vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eTransferRead,
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eTransfer);

    return imageBuffer;
}

void fill(void* memory, int width, int height, int depth)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    ImageView view(width, height, depth, memory);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint8_t tmp = static_cast<uint8_t>(dis(gen));
            view(x, y, 0) = tmp;
            view(x, y, 1) = tmp;
            view(x, y, 2) = tmp;
            view(x, y, 3) = 0xFF;
        }
    }
}

std::pair<vk::UniqueImage, vk::UniqueDeviceMemory>
createHostVisibleTextureImage(const vk::PhysicalDevice& physicalDevice,
                              const vk::UniqueDevice& device,
                              const vk::Queue& queue,
                              const vk::UniqueCommandPool& commandPool)
{
    auto imageBuffer = createImage(
        physicalDevice, device, WIDTH, HEIGHT, vk::Format::eR8G8B8A8Unorm,
        vk::ImageLayout::ePreinitialized, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent);

    return imageBuffer;
}

void updateUbo(const vk::UniqueDevice& device,
               const vk::UniqueDeviceMemory& memory,
               const std::array<float, 3>& color)
{
    void* data = device->mapMemory(memory.get(), 0, color.size() * 3);
    std::memcpy(data, color.data(), color.size() * 3);
    device->unmapMemory(memory.get());
}

void copyImage(const vk::UniqueDevice& device, const vk::Queue& queue,
               const vk::UniqueCommandPool& commandPool, const vk::Image& src,
               const vk::Image& dst)
{

    vk::CommandBufferAllocateInfo allocInfo;
    allocInfo.setCommandPool(commandPool.get());
    allocInfo.setLevel(vk::CommandBufferLevel::ePrimary);
    allocInfo.setCommandBufferCount(1);
    auto command = device->allocateCommandBuffersUnique(allocInfo);

    transitionImageLayout(
        device, queue, commandPool, src, vk::Format::eR8G8B8A8Unorm,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferSrcOptimal,
        vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eTransferRead,
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eTransfer);

    transitionImageLayout(
        device, queue, commandPool, dst, vk::Format::eR8G8B8A8Unorm,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
        vk::AccessFlagBits::eMemoryRead, vk::AccessFlagBits::eTransferWrite,
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eTransfer);

    vk::CommandBufferBeginInfo beginInfo;
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    command.front()->begin(beginInfo);

    vk::ImageCopy copyRegion;
    copyRegion.setSrcSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1});
    copyRegion.setSrcOffset({0, 0, 0});
    copyRegion.setDstSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1});
    copyRegion.setDstOffset({0, 0, 0});
    copyRegion.setExtent({WIDTH, HEIGHT, 1});
    command.front()->copyImage(src, vk::ImageLayout::eTransferSrcOptimal, dst,
                               vk::ImageLayout::eTransferDstOptimal, 1,
                               &copyRegion);
    command.front()->end();
    vk::SubmitInfo submitInfo;
    submitInfo.setCommandBufferCount(1);
    submitInfo.setPCommandBuffers(&command.front().get());
    queue.submit(1, &submitInfo, vk::Fence{});
    queue.waitIdle();
    transitionImageLayout(
        device, queue, commandPool, dst, vk::Format::eR8G8B8A8Unorm,
        vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR,
        vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eMemoryRead,
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eBottomOfPipe);
}

int main()
{
    glfwInit();

    auto availableExtensions = vk::enumerateInstanceExtensionProperties();
    auto requiredExtensions = getRequiredExtensionsForGlfw();

    if (meetExtensionRequirements(availableExtensions, requiredExtensions)) {
        // Create vulkan instance
        std::cout << "Create Instance" << std::endl;
        auto instance = createInstance(requiredExtensions);

        // Load debug messenger extension and register callbacks for
        // layer validation
        vk::DispatchLoaderDynamic dldi(instance.get());
        std::cout << "Create DebugMessagenger" << std::endl;
        auto messanger = createDebugMessenger(instance, dldi);

        // Create window using glfw
        std::cout << "Create window" << std::endl;
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        auto window = glfw::createWindow(WIDTH, HEIGHT, "test");

        // Retrieve vulkan surface from windwo
        auto surface = window.createWindowSurface(instance);

        // Use physical device
        auto physicalDevice = pickDevice(instance);

        // Check if physical device has swap chain capabilities
        // (i.e. presentation mode), which are needed for rendering.
        auto swapChainCapabilities =
            querySwapChainSupport(physicalDevice, surface);

        if (swapChainCapabilities.formats.empty() ||
            swapChainCapabilities.presentModes.empty()) {
            throw std::runtime_error("Swap chain requirements not meet");
        }

        auto surfaceFormat =
            chooseSwapSurfaceFormat(swapChainCapabilities.formats);

        auto presentMode =
            chooseSwapPresentMode(swapChainCapabilities.presentModes);

        auto extent =
            chooseSwapExtent(swapChainCapabilities.capabilities, WIDTH, HEIGHT);

        printSurfaceCapabilities(swapChainCapabilities.capabilities);
        auto queueFamilyIndex = findQueueFamilies(physicalDevice);

        if (!queueFamilyIndex.has_value()) {
            throw std::runtime_error("No suitable queue family found");
        }

        // Test if queue of device supports presentation
        if (!physicalDevice.getSurfaceSupportKHR(*queueFamilyIndex,
                                                 surface.get())) {
            throw std::runtime_error("Queue does not support presentation");
        }

        // Create device from physical device which was selected previously
        auto device = createDevice(physicalDevice, *queueFamilyIndex);

        // Create swap chain from surface
        auto swapChain = createSwapChain(device, surface, extent, presentMode,
                                         surfaceFormat);

        auto swapChainImages = device->getSwapchainImagesKHR(swapChain.get());
        auto swapChainImageViews = createImageViewsFromSwapChain(
            device, swapChainImages, surfaceFormat);
        auto queue = device->getQueue(*queueFamilyIndex, 0);

        // Create Descriptor Sets
        vk::DescriptorPoolSize poolSize;
        poolSize.setType(vk::DescriptorType::eUniformBuffer);
        poolSize.setDescriptorCount(
            static_cast<uint32_t>(swapChainImages.size()));
        vk::DescriptorPoolCreateInfo descPoolInfo;
        descPoolInfo.setMaxSets(static_cast<uint32_t>(swapChainImages.size()));
        descPoolInfo.setPoolSizeCount(1);
        descPoolInfo.setFlags(
            vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
        descPoolInfo.setPPoolSizes(&poolSize);

        auto descriptorPool = device->createDescriptorPoolUnique(descPoolInfo);

        auto descriptorSetLayouts =
            createDescriptorSetLayouts(device, swapChainImages.size());
        vk::DescriptorSetAllocateInfo allocateInfo;
        allocateInfo.setDescriptorPool(descriptorPool.get());
        allocateInfo.setDescriptorSetCount(
            static_cast<uint32_t>(swapChainImages.size()));
        allocateInfo.setPSetLayouts(descriptorSetLayouts.data());
        auto descriptorSet = device->allocateDescriptorSets(allocateInfo);

        auto [renderPass, graphicPipeline, pipelineLayout] =
            createPipeline(device, descriptorSetLayouts, extent, surfaceFormat);

        auto frameBuffers =
            createFrameBuffers(device, renderPass, swapChainImageViews, extent);

        auto commandPool = createCommandPool(device, *queueFamilyIndex);

        auto texture =
            createTextureImage(physicalDevice, device, queue, commandPool);

        auto visibleTexture = createHostVisibleTextureImage(
            physicalDevice, device, queue, commandPool);

        void* map = device->mapMemory(visibleTexture.second.get(), 0,
                                      WIDTH * HEIGHT * 4);

        fill(map, WIDTH, HEIGHT, 4);
        device->unmapMemory(visibleTexture.second.get());

        auto [renderFinished, imageAvailable] = createSemaphores(device);

        std::vector<std::pair<vk::UniqueBuffer, vk::UniqueDeviceMemory>>
            buffers;

        for (const auto& iter : swapChainImages) {
            buffers.push_back(
                createBuffer(physicalDevice, device, sizeof(float) * 3,
                             vk::BufferUsageFlagBits::eUniformBuffer,
                             vk::MemoryPropertyFlagBits::eHostVisible |
                                 vk::MemoryPropertyFlagBits::eHostCoherent));
        }

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            vk::DescriptorBufferInfo bufferInfo;
            bufferInfo.setBuffer(buffers[i].first.get());
            bufferInfo.setRange(VK_WHOLE_SIZE);
            bufferInfo.setOffset(0);

            vk::WriteDescriptorSet writeDesc;
            writeDesc.setDstSet(descriptorSet[i]);
            writeDesc.setDstBinding(0);
            writeDesc.setDstArrayElement(0);
            writeDesc.setDescriptorCount(1);
            writeDesc.setDescriptorType(vk::DescriptorType::eUniformBuffer);
            writeDesc.setPBufferInfo(&bufferInfo);
            device->updateDescriptorSets(1, &writeDesc, 0, nullptr);
        }

        auto commandBuffers = createDrawCommand(
            device, commandPool, graphicPipeline, renderPass, frameBuffers,
            pipelineLayout, descriptorSet, extent);

        auto colors = std::array<std::array<float, 3>, 3>{
            {std::array<float, 3>{1.0f, 0.0f, 0.0f},
             std::array<float, 3>{0.0f, 1.0f, 0.0f},
             std::array<float, 3>{1.0f, 0.0f, 0.0f}}};

        // Draw
        vk::FenceCreateInfo fenceInfo;
        auto fence = device->createFence(fenceInfo);

        Timer<std::chrono::milliseconds> timer;
        int counter = 0;

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            uint32_t index = 0;
            device->acquireNextImageKHR(swapChain.get(),
                                        std::numeric_limits<uint64_t>::max(),
                                        vk::Semaphore{}, fence, &index);
            device->waitForFences(1, &fence, true,
                                  std::numeric_limits<uint64_t>::max());
            device->resetFences(1, &fence);

            void* map = device->mapMemory(visibleTexture.second.get(), 0,
                                          WIDTH * HEIGHT * 4);

            fill(map, WIDTH, HEIGHT, 4);
            device->unmapMemory(visibleTexture.second.get());

            copyImage(device, queue, commandPool, visibleTexture.first.get(),
                      swapChainImages[index]);

            transitionImageLayout(
                device, queue, commandPool, visibleTexture.first.get(),
                vk::Format::eR8G8B8A8Unorm,
                vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eGeneral,
                vk::AccessFlagBits::eTransferRead,
                vk::AccessFlagBits::eTransferRead,
                vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eTransfer);

            // updateUbo(device, buffers[index].second, colors[index]);
            // vk::SubmitInfo submitInfo;
            // vk::PipelineStageFlags waitStages[] = {
            //     vk::PipelineStageFlagBits::eColorAttachmentOutput};
            // submitInfo.setWaitSemaphoreCount(1);
            // submitInfo.setPWaitSemaphores(&imageAvailable.get());
            // submitInfo.setPWaitDstStageMask(waitStages);
            // submitInfo.setCommandBufferCount(1);
            // submitInfo.setPCommandBuffers(&commandBuffers[index]);
            // submitInfo.setSignalSemaphoreCount(1);
            // submitInfo.setPSignalSemaphores(&renderFinished.get());
            // queue.submit(1, &submitInfo, vk::Fence{});

            vk::PresentInfoKHR presentInfo;
            presentInfo.setWaitSemaphoreCount(0);
            presentInfo.setPWaitSemaphores(nullptr);
            presentInfo.setSwapchainCount(1);
            presentInfo.setPSwapchains(&swapChain.get());
            presentInfo.setPImageIndices(&index);
            queue.presentKHR(presentInfo);

            if (timer.elapsed() < 1000ms) {
                counter++;
            } else {
                std::cout << "fps: " << counter << std::endl;
                timer.reset();
                counter = 0;
            }
        }
    }

    std::cout << "Glfw extensions" << std::endl;

    return 0;
}
