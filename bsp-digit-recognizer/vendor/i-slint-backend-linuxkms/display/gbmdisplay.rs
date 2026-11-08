// Copyright © SixtyFPS GmbH <info@slint.dev>
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Slint-Royalty-free-2.0 OR LicenseRef-Slint-Software-3.0

// cSpell: ignore SCANOUT
use drm::control::Device;
use gbm::AsRaw;
use glutin::config::{ColorBufferType, GlConfig};
use std::cell::Cell;
use i_slint_core::platform::PlatformError;

use crate::drmoutput::{DrmOutput, SharedFd};

struct OwnedFramebufferHandle {
    handle: drm::control::framebuffer::Handle,
    device: SharedFd,
}

impl Drop for OwnedFramebufferHandle {
    fn drop(&mut self) {
        self.device.destroy_framebuffer(self.handle).ok();
    }
}

pub struct GbmDisplay {
    pub drm_output: DrmOutput,
    gbm_surface: gbm::Surface<OwnedFramebufferHandle>,
    gbm_device: gbm::Device<SharedFd>,
    surface_format: drm::buffer::DrmFourcc,
    logged_frame: Cell<bool>,
}

impl GbmDisplay {
    pub fn new(drm_output: DrmOutput) -> Result<GbmDisplay, PlatformError> {
        //eprintln!("mode {}/{}", width, height);

        let gbm_device = gbm::Device::new(drm_output.drm_device.clone())
            .map_err(|e| format!("Error creating gbm device: {e}"))?;

        // DRM AR24 is the little-endian 0xAARRGGBB scanout format. Keep alpha
        // from the FemtoVG canvas through EGL, GBM and the VOP2 AFBC decoder.
        let surface_format = gbm::Format::Argb8888;

        let (width, height) = drm_output.size();
        let modifiers = drm_output.primary_modifiers(surface_format)?;
        let gbm_surface = gbm_device
            .create_surface_with_modifiers2::<OwnedFramebufferHandle>(
                width,
                height,
                surface_format,
                modifiers.into_iter().map(gbm::Modifier::from),
                gbm::BufferObjectFlags::SCANOUT | gbm::BufferObjectFlags::RENDERING,
            )
            .map_err(|e| format!("Error creating gbm surface: {e}"))?;

        Ok(GbmDisplay { drm_output, gbm_surface, gbm_device, surface_format, logged_frame: Cell::new(false) })
    }

    pub fn config_template_builder(&self) -> glutin::config::ConfigTemplateBuilder {
        glutin::config::ConfigTemplateBuilder::new()
            .with_buffer_type(ColorBufferType::Rgb { r_size: 8, g_size: 8, b_size: 8 })
            .with_alpha_size(8)
            .with_transparency(true)
    }

    pub fn filter_gl_config(&self, config: &glutin::config::Config) -> bool {
        #[allow(unreachable_patterns)] // GBM builds currently provide only EGL configs.
        match &config {
            glutin::config::Config::Egl(egl_config) => {
                drm::buffer::DrmFourcc::try_from(egl_config.native_visual())
                    == Ok(self.surface_format)
                    && config.alpha_size() == 8
                    && config.color_buffer_type()
                        == Some(ColorBufferType::Rgb { r_size: 8, g_size: 8, b_size: 8 })
            }
            _ => false,
        }
    }
}

impl super::Presenter for GbmDisplay {
    fn present(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut front_buffer = unsafe {
            self.gbm_surface
                .lock_front_buffer()
                .map_err(|err| format!("Could not lock gbm front buffer: {err}"))?
        };

        if drm::buffer::Buffer::format(&front_buffer) != self.surface_format {
            return Err("GBM returned a framebuffer that is not ARGB8888".into());
        }

        // Same logic as in drm-rs' `add_planar_framebuffer` function.
        let flags = if drm::buffer::PlanarBuffer::modifier(&front_buffer)
            .filter(|modifier| !matches!(modifier, drm::buffer::DrmModifier::Invalid))
            .is_some()
        {
            drm::control::FbCmd2Flags::MODIFIERS
        } else {
            drm::control::FbCmd2Flags::empty()
        };

        let fb = self
            .drm_output
            .drm_device
            .add_planar_framebuffer(&front_buffer, flags)
            .map_err(|e| format!("Error adding gbm buffer as framebuffer: {e}"))?;

        if !self.logged_frame.replace(true) {
            eprintln!("GPU_SCANOUT: framebuffer={fb:?}; format={:?}; modifier={:?}; pitch={}",
                drm::buffer::Buffer::format(&front_buffer),
                drm::buffer::PlanarBuffer::modifier(&front_buffer),
                drm::buffer::Buffer::pitch(&front_buffer));
        }

        front_buffer.set_userdata(OwnedFramebufferHandle {
            handle: fb,
            device: self.drm_output.drm_device.clone(),
        });

        self.drm_output.present(front_buffer, fb)
    }
}

impl raw_window_handle::HasWindowHandle for GbmDisplay {
    fn window_handle(
        &self,
    ) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
        Ok(unsafe {
            let gbm_surface_handle = raw_window_handle::GbmWindowHandle::new(
                std::ptr::NonNull::from(&*self.gbm_surface.as_raw()).cast(),
            );

            raw_window_handle::WindowHandle::borrow_raw(raw_window_handle::RawWindowHandle::Gbm(
                gbm_surface_handle,
            ))
        })
    }
}

impl raw_window_handle::HasDisplayHandle for GbmDisplay {
    fn display_handle(
        &self,
    ) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
        Ok(unsafe {
            let gbm_display_handle = raw_window_handle::GbmDisplayHandle::new(
                std::ptr::NonNull::from(&*self.gbm_device.as_raw()).cast(),
            );

            raw_window_handle::DisplayHandle::borrow_raw(raw_window_handle::RawDisplayHandle::Gbm(
                gbm_display_handle,
            ))
        })
    }
}
