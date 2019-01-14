#include <vulkan/vulkan.h>
#include <vulkan/vulkan.hpp>
#include <glfw.hpp>
#include <GLFW/glfw3.h>
#include <fstream>

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
    bool available = false;

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
    info.setImageUsage(vk::ImageUsageFlagBits::eColorAttachment);
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

vk::UniqueDescriptorSetLayout
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

    return device->createDescriptorSetLayoutUnique(descriptorSetCreateInfo);
}
vk::UniquePipelineLayout createPipelineLayout(const vk::UniqueDevice& device)
{

    auto descriptor = createDescriptorSetLayout(device);
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    pipelineLayoutInfo.setSetLayoutCount(1);
    pipelineLayoutInfo.setPSetLayouts(&descriptor.get());

    return device->createPipelineLayoutUnique(pipelineLayoutInfo);
}

std::pair<vk::UniqueRenderPass, vk::UniquePipeline>
createPipeline(const vk::UniqueDevice& device, const vk::Extent2D& extent,
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
    pipelineLayoutInfo.setSetLayoutCount(0);
    pipelineLayoutInfo.setPushConstantRangeCount(0);

    auto pipelineLayout = createPipelineLayout(device);
    // auto pipelineLayout =
    //     device->createPipelineLayoutUnique(pipelineLayoutInfo);

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
    return std::make_pair(std::move(renderPass), std::move(pipeline));
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
    device->createCommandPoolUnique(commandPoolInfo);
    return device->createCommandPoolUnique(commandPoolInfo);
}

std::vector<vk::CommandBuffer> createCommandBuffers(
    const vk::UniqueDevice& device, const vk::UniqueCommandPool& commandPool,
    const vk::UniquePipeline& pipeline, const vk::UniqueRenderPass& renderPass,
    const std::vector<vk::UniqueFramebuffer>& swapChainFrameBuffers,
    const vk::Extent2D& extent)
{
    // std::vector<vk::CommandBuffer>
    // commandBuffers(swapChainFrameBuffers.size());

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
        auto window = glfw::createWindow(800, 600, "test");

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
            chooseSwapExtent(swapChainCapabilities.capabilities, 800, 600);

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

        auto [renderPass, graphicPipeline] =
            createPipeline(device, extent, surfaceFormat);

        auto frameBuffers =
            createFrameBuffers(device, renderPass, swapChainImageViews, extent);

        auto commandPool = createCommandPool(device, *queueFamilyIndex);

        auto commandBuffers =
            createCommandBuffers(device, commandPool, graphicPipeline,
                                 renderPass, frameBuffers, extent);

        auto [renderFinished, imageAvailable] = createSemaphores(device);

        // Descriptor Sets

        vk::DescriptorPoolSize poolSize;
        poolSize.setType(vk::DescriptorType::eUniformBuffer);
        poolSize.setDescriptorCount(2);
        vk::DescriptorPoolCreateInfo descPoolInfo;
        descPoolInfo.setMaxSets(10);
        descPoolInfo.setPoolSizeCount(1);
        descPoolInfo.setFlags(
            vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
        descPoolInfo.setPPoolSizes(&poolSize);

        auto descriptorPool = device->createDescriptorPoolUnique(descPoolInfo);

        auto descriptorSetLayout = createDescriptorSetLayout(device);
        vk::DescriptorSetAllocateInfo allocateInfo;
        allocateInfo.setDescriptorPool(descriptorPool.get());
        allocateInfo.setDescriptorSetCount(1);
        allocateInfo.setPSetLayouts(&descriptorSetLayout.get());
        auto descriptorSet = device->allocateDescriptorSetsUnique(allocateInfo);

        vk::BufferCreateInfo bufferCreateInfo;
        bufferCreateInfo.setSize(sizeof(float) * 3);
        bufferCreateInfo.setUsage(vk::BufferUsageFlagBits::eUniformBuffer);
        bufferCreateInfo.setSharingMode(vk::SharingMode::eExclusive);

        auto buffer = device->createBufferUnique(bufferCreateInfo);

        vk::DescriptorBufferInfo bufferInfo;
        bufferInfo.setBuffer(buffer.get());
        bufferInfo.setRange(VK_WHOLE_SIZE);
        bufferInfo.setOffset(0);

        vk::WriteDescriptorSet writeDesc;
        writeDesc.setDstSet(descriptorSet[0].get());
        writeDesc.setDstBinding(0);
        writeDesc.setDstArrayElement(0);
        writeDesc.setDescriptorCount(1);
        writeDesc.setDescriptorType(vk::DescriptorType::eUniformBuffer);
        writeDesc.setPBufferInfo(&bufferInfo);

        // Draw
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            uint32_t index = 0;
            device->acquireNextImageKHR(
                swapChain.get(), std::numeric_limits<uint64_t>::max(),
                imageAvailable.get(), vk::Fence{}, &index);

            vk::SubmitInfo submitInfo;
            vk::PipelineStageFlags waitStages[] = {
                vk::PipelineStageFlagBits::eColorAttachmentOutput};
            submitInfo.setWaitSemaphoreCount(1);
            submitInfo.setPWaitSemaphores(&imageAvailable.get());
            submitInfo.setPWaitDstStageMask(waitStages);
            submitInfo.setCommandBufferCount(1);
            submitInfo.setPCommandBuffers(&commandBuffers[index]);
            submitInfo.setSignalSemaphoreCount(1);
            submitInfo.setPSignalSemaphores(&renderFinished.get());
            queue.submit(1, &submitInfo, vk::Fence{});

            vk::PresentInfoKHR presentInfo;
            presentInfo.setWaitSemaphoreCount(1);
            presentInfo.setPWaitSemaphores(&renderFinished.get());
            presentInfo.setSwapchainCount(1);
            presentInfo.setPSwapchains(&swapChain.get());
            presentInfo.setPImageIndices(&index);
            queue.presentKHR(presentInfo);
        }
        //  std::cout << index. << std::endl;

        // uint32_t imageIndex = device->acquireNextImageKHR(
        //     swapChain.get(), std::numeric_limits<uint64_t>::max(),
        //     imageAvailable, nullptr);

        // std::vector<vk::UniqueCommandBuffer> commandBuffers;
    }

    std::cout << "Glfw extensions" << std::endl;

    //  initVulkan();
    // uint32_t extensionCount = 0;

    // auto result = vk::enumerateInstanceLayerProperties();
    // std::cout << result.size() << std::endl;

    // glfwInit();

    // vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount,
    // nullptr);

    // std::cout << extensionCount << " extensions supported" << std::endl;

    // // glm::mat4 matrix;
    // // glm::vec4 vec;
    // // auto test = matrix * vec;

    // while (!glfwWindowShouldClose(window)) {
    //     glfwPollEvents();
    // }

    // glfwDestroyWindow(window);

    // glfwTerminate();

    return 0;
}
