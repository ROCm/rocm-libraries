import amd.rocal.fn as fn
import amd.rocal.types as types
import numpy as np

#batch size = 64
rocalList_mode1_64 = ['original', 'warpAffine', 'contrast', 'rain', 
            'brightness', 'colorTemp', 'exposure', 'vignette', 
            'blur', 'snow', 'pixelate', 'SnPNoise', 
            'gamma', 'rotate', 'flip', 'blend',
            'rotate45+resize', 'rotate45+warpAffine', 'rotate45+contrast', 'rotate45+rain', 
            'rotate45+brightness', 'rotate45+colorTemp', 'rotate45+exposure', 'rotate45+vignette', 
            'rotate45+blur', 'rotate45+snow', 'rotate45+pixelate', 'rotate45+SnPNoise', 
            'rotate45+gamma', 'rotate45+rotate', 'rotate45+flip', 'rotate45+blend',
            'flip+resize', 'flip+warpAffine', 'flip+contrast', 'flip+rain', 
            'flip+brightness', 'flip+colorTemp', 'flip+exposure', 'flip+vignette', 
            'flip+blur', 'flip+snow', 'flip+pixelate', 'flip+SnPNoise', 
            'flip+gamma', 'flip+rotate', 'flip+flip', 'flip+blend',			
            'rotate150+resize', 'rotate150+warpAffine', 'rotate150+contrast', 'rotate150+rain', 
            'rotate150+brightness', 'rotate150+colorTemp', 'rotate150+exposure', 'rotate150+vignette', 
            'rotate150+blur', 'rotate150+snow', 'rotate150+pixelate', 'rotate150+SnPNoise', 
            'rotate150+gamma', 'rotate150+rotate', 'rotate150+flip', 'rotate150+blend']
rocalList_mode2_64 = ['original', 'warpAffine', 'contrast', 'rain', 
            'brightness', 'colorTemp', 'exposure', 'vignette', 
            'blur', 'snow', 'pixelate', 'SnPNoise', 
            'gamma', 'rotate', 'flip', 'blend',
            'warpAffine+original', 'warpAffine+warpAffine', 'warpAffine+contrast', 'warpAffine+rain', 
            'warpAffine+brightness', 'warpAffine+colorTemp', 'warpAffine+exposure', 'warpAffine+vignette', 
            'warpAffine+blur', 'warpAffine+snow', 'pixelate', 'warpAffine+SnPNoise', 
            'warpAffine+gamma', 'warpAffine+rotate', 'warpAffine+flip', 'warpAffine+blend',
            'fishEye+original', 'fishEye+warpAffine', 'fishEye+contrast', 'fishEye+rain', 
            'fishEye+brightness', 'fishEye+colorTemp', 'fishEye+exposure', 'fishEye+vignette', 
            'fishEye+blur', 'fishEye+snow', 'fishEye+pixelate', 'fishEye+SnPNoise', 
            'fishEye+gamma', 'fishEye+rotate', 'fishEye+flip', 'fishEye+blend',
            'lensCorrection+original', 'lensCorrection+warpAffine', 'lensCorrection+contrast', 'lensCorrection+rain', 
            'lensCorrection+brightness', 'lensCorrection+colorTemp', 'exposure', 'lensCorrection+vignette', 
            'lensCorrection+blur', 'lensCorrection+snow', 'lensCorrection+pixelate', 'lensCorrection+SnPNoise', 
            'lensCorrection+gamma', 'lensCorrection+rotate', 'lensCorrection+flip', 'lensCorrection+blend',]
rocalList_mode3_64 = ['original', 'warpAffine', 'contrast', 'rain', 
            'brightness', 'colorTemp', 'exposure', 'vignette', 
            'blur', 'snow', 'pixelate', 'SnPNoise', 
            'gamma', 'rotate', 'flip', 'blend',
            'colorTemp+original', 'colorTemp+warpAffine', 'colorTemp+contrast', 'colorTemp+rain', 
            'colorTemp+brightness', 'colorTemp+colorTemp', 'colorTemp+exposure', 'colorTemp+vignette', 
            'colorTemp+blur', 'colorTemp+snow', 'colorTemp+pixelate', 'colorTemp+SnPNoise', 
            'colorTemp+gamma', 'colorTemp+rotate', 'colorTemp+flip', 'colorTemp+blend',
            'colorTemp+original', 'colorTemp+warpAffine', 'colorTemp+contrast', 'colorTemp+rain', 
            'colorTemp+brightness', 'colorTemp+colorTemp', 'colorTemp+exposure', 'colorTemp+vignette', 
            'colorTemp+blur', 'colorTemp+snow', 'colorTemp+pixelate', 'colorTemp+SnPNoise', 
            'colorTemp+gamma', 'colorTemp+rotate', 'colorTemp+flip', 'colorTemp+blend',
            'warpAffine+original', 'warpAffine+warpAffine', 'warpAffine+contrast', 'warpAffine+rain', 
            'warpAffine+brightness', 'warpAffine+colorTemp', 'warpAffine+exposure', 'warpAffine+vignette', 
            'warpAffine+blur', 'warpAffine+snow', 'pixelate', 'warpAffine+SnPNoise', 
            'warpAffine+gamma', 'warpAffine+rotate', 'warpAffine+flip', 'warpAffine+blend']
rocalList_mode4_64 = ['original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original']
rocalList_mode5_64 = ['original', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop']
#batch size = 16
rocalList_mode1_16 = ['original', 'warpAffine', 'contrast', 'rain', 
            'brightness', 'colorTemp', 'exposure', 'vignette', 
            'blur', 'snow', 'pixelate', 'SnPNoise', 
            'gamma', 'rotate', 'flip', 'blend']
rocalList_mode2_16 = ['original', 'warpAffine', 'contrast', 'contrast+rain', 
            'brightness', 'brightness+colorTemp', 'exposure', 'exposure+vignette', 
            'blur', 'blur+snow', 'pixelate', 'pixelate+SnPNoise', 
            'gamma', 'rotate', 'rotate+flip', 'blend']
rocalList_mode3_16 = ['original', 'warpAffine', 'contrast', 'warpAffine+rain', 
            'brightness', 'colorTemp', 'exposure', 'vignette', 
            'blur', 'vignette+snow', 'pixelate', 'gamma',
            'SnPNoise+gamma', 'rotate', 'flip+pixelate', 'blend']
rocalList_mode4_16 = ['original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original']
rocalList_mode5_16 = ['original', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop']

# Class to initialize rocal and call the augmentations 
class InferencePipe():
    def __init__(self, pipe, image_validation, model_batch_size, rocalMode, c_img, h_img, w_img, rocal_batch_size, tensor_dtype, multiplier, offset, tensor_layout, num_threads, device_id, data_dir, crop, rocal_cpu = True):
        rocal_device = 'cpu' if rocal_cpu else 'gpu'
        decoder_device = 'cpu' if rocal_cpu else 'mixed'
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        
        self.pipe = pipe
        self.validation_dict = {}
        self.process_validation(image_validation)
        self.pipe.set_seed(0)
        self.aug_strength = 0
        self.model_batch_size = model_batch_size
        self.rocalMode =  rocalMode
        self.data_dir = data_dir
        self.c_img = c_img
        self.h_img = h_img
        self.w_img = w_img
        self.rocal_batch_size = rocal_batch_size
        self.tensor_dtype = tensor_dtype
        self.multiplier = multiplier
        self.offset = offset
        self.tensor_layout = tensor_layout
        self.reverse_channels = False
        #for tensor output
        self.bs = self.rocal_batch_size
        if self.tensor_dtype == types.FLOAT:
            self.out_tensor = np.zeros(( self.bs*self.model_batch_size, self.c_img, int(self.h_img/self.bs), self.w_img,), dtype = "float32")
        elif self.tensor_dtype == types.FLOAT16:
            self.out_tensor = np.zeros(( self.bs*self.model_batch_size, self.c_img, int(self.h_img/self.bs), self.w_img,), dtype = "float16")
        
        self.random_shuffle = True
        self.shard_id = 0
        self.num_shards = 1

        #params for contrast
        self.min_param = self.pipe.create_int_param(0)
        self.max_param = self.pipe.create_int_param(255)
        #param for brightness
        self.alpha_param = self.pipe.create_float_param(1.0)
        self.beta_param = self.pipe.create_float_param(10)
        #param for ColorTemperature     
        self.adjustment_param = self.pipe.create_int_param(0)
        self.adjustment_param_10 = self.pipe.create_int_param(10)
        self.adjustment_param_20 = self.pipe.create_int_param(20)
        #param for exposure
        self.shift_param = self.pipe.create_float_param(0.0)
        #param for SnPNoise
        self.sdev_param = self.pipe.create_float_param(0.0)
        #param for gamma
        self.gamma_shift_param = self.pipe.create_float_param(0.0)
        #param for rotate
        self.degree_param = self.pipe.create_float_param(0.0)
        self.degree_param_150 = self.pipe.create_float_param(150.0)
        self.degree_param_45 = self.pipe.create_float_param(45.0)
        #param for lens correction
        self.strength_param = self.pipe.create_float_param(1.5)
        self.zoom_param = self.pipe.create_float_param(2.0)
        #params for flip
        self.flip_param = self.pipe.create_int_param(1)
        #param for snow
        self.snow_param = self.pipe.create_float_param(0.1)
        #param for rain
        self.rain_param = self.pipe.create_float_param(0.1)
        self.rain_width_param = self.pipe.create_int_param(2)
        self.rain_height_param = self.pipe.create_int_param(15)
        self.rain_transparency_param = self.pipe.create_float_param(0.25)
        #param for blur
        self.blur_param = self.pipe.create_int_param(5)
        #param for jitter
        self.kernel_size = self.pipe.create_int_param(3)
        #param for warp affine
        self.affine_matrix_param = [0.35,0.25,0.75,1,1,1]
        self.affine_matrix_1_param = [0.5, 0 , 0, 2, 1, 1]
        self.affine_matrix_2_param = [2, 0, 0, 1, 1, 1]
        #param for vignette
        self.vignette_param = self.pipe.create_float_param(50)
        #param for blend
        self.blend_param = self.pipe.create_float_param(0.5)

        #rocal list of augmentation
        self.rocal_list = None

        #read file and decode images - common for all rocal modes
        with self.pipe:
            self.jpegs, self.labels = fn.readers.file(file_root=self.data_dir)
            self.decoded_images = fn.decoders.image(self.jpegs, device=decoder_device, output_type=types.RGB,
                                                device_memory_padding=device_memory_padding,
                                                host_memory_padding=host_memory_padding,
                                                file_root=self.data_dir, shard_id=self.shard_id, num_shards=self.num_shards, random_shuffle=self.random_shuffle)
            self.input = fn.resize(self.decoded_images, device=rocal_device, resize_x=crop, resize_y=crop)
            if model_batch_size == 16:
                if rocalMode == 1:
                    self.warped = fn.warp_affine(self.input, matrix=self.affine_matrix_param)
                    self.contrast_img = fn.contrast(self.input, min_contrast=self.min_param, max_contrast=self.max_param)
                    self.rain_img = fn.rain(self.input, rain=self.rain_param, rain_width = self.rain_width_param, rain_height = self.rain_height_param, rain_transparency =self.rain_transparency_param)
                    self.bright_img = fn.brightness_fixed(self.input, alpha=self.alpha_param, beta= self.beta_param)
                    self.temp_img = fn.color_temp(self.input, adjustment_value=self.adjustment_param)
                    self.exposed_img = fn.exposure(self.input, exposure=self.shift_param)
                    self.vignette_img = fn.vignette(self.input, vignette = self.vignette_param)
                    self.blur_img = fn.blur(self.input, blur = self.blur_param)
                    self.snow_img = fn.snow(self.input, snow=self.snow_param)
                    self.pixelate_img = fn.pixelate(self.input)
                    self.snp_img = fn.snp_noise(self.input, snpNoise=self.sdev_param)
                    self.gamma_img = fn.gamma_correction(self.input, gamma=self.gamma_shift_param)
                    self.rotate_img = fn.rotate(self.input, angle=self.degree_param)
                    self.flip_img = fn.flip(self.input, flip=self.flip_param)
                    self.blend_img = fn.blend(self.input, self.contrast_img)

                    # set outputs for this mode
                    self.pipe.set_outputs(self.input, self.warped, self.contrast_img, self.rain_img, self.bright_img,
                                    self.temp_img, self.exposed_img, self.vignette_img, self.blur_img, self.snow_img,
                                    self.pixelate_img, self.snp_img, self.gamma_img, self.rotate_img, self.flip_img, self.blend_img)
                elif rocalMode == 2:
                    self.warped = fn.warp_affine(self.input, matrix=self.affine_matrix_param)
                    self.contrast_img = fn.contrast(self.input, min_contrast=self.min_param, max_contrast=self.max_param)
                    self.rain_img = fn.rain(self.contrast_img, rain=self.rain_param, rain_width = self.rain_width_param, rain_height = self.rain_height_param, rain_transparency =self.rain_transparency_param)
                    self.bright_img = fn.brightness_fixed(self.input, alpha=self.alpha_param, beta= self.beta_param)
                    self.temp_img = fn.color_temp(self.bright_img, adjustment_value=self.adjustment_param)
                    self.exposed_img = fn.exposure(self.input, exposure=self.shift_param)
                    self.vignette_img = fn.vignette(self.exposed_img, vignette = self.vignette_param)
                    self.blur_img = fn.blur(self.input, blur = self.blur_param)
                    self.snow_img = fn.snow(self.blur_img, snow=self.snow_param)
                    self.pixelate_img = fn.pixelate(self.input)
                    self.snp_img = fn.snp_noise(self.pixelate_img, snpNoise=self.sdev_param)
                    self.gamma_img = fn.gamma_correction(self.input, gamma=self.gamma_shift_param)
                    self.rotate_img = fn.rotate(self.input, angle=self.degree_param)
                    self.flip_img = fn.flip(self.rotate_img, flip=self.flip_param)
                    self.blend_img = fn.blend(self.input, self.contrast_img)

                    # set outputs for this mode
                    self.pipe.set_outputs(self.input, self.warped, self.contrast_img, self.rain_img, self.bright_img,
                                    self.temp_img, self.exposed_img, self.vignette_img, self.blur_img, self.snow_img,
                                    self.pixelate_img, self.snp_img, self.gamma_img, self.rotate_img, self.flip_img, self.blend_img)
                elif rocalMode == 3:
                    self.warped = fn.warp_affine(self.input, matrix=self.affine_matrix_param)
                    self.contrast_img = fn.contrast(self.input, min_contrast=self.min_param, max_contrast=self.max_param)
                    self.rain_img = fn.rain(self.warped, rain=self.rain_param, rain_width = self.rain_width_param, rain_height = self.rain_height_param, rain_transparency =self.rain_transparency_param)
                    self.bright_img = fn.brightness_fixed(self.input, alpha=self.alpha_param, beta= self.beta_param)
                    self.temp_img = fn.color_temp(self.input, adjustment_value=self.adjustment_param)
                    self.exposed_img = fn.exposure(self.input, exposure=self.shift_param)
                    self.vignette_img = fn.vignette(self.input, vignette = self.vignette_param)
                    self.blur_img = fn.blur(self.input, blur = self.blur_param)
                    self.snow_img = fn.snow(self.vignette_img, snow=self.snow_param)
                    self.pixelate_img = fn.pixelate(self.input)
                    self.gamma_img = fn.gamma_correction(self.input, gamma=self.gamma_shift_param)
                    self.snp_img = fn.snp_noise(self.gamma_img, snpNoise=self.sdev_param)
                    self.rotate_img = fn.rotate(self.input, angle=self.degree_param)
                    self.flip_img = fn.flip(self.pixelate_img, flip=self.flip_param)
                    self.blend_img = fn.blend(self.input, self.contrast_img)
                    
                    # set outputs for this mode
                    self.pipe.set_outputs(self.input, self.warped, self.contrast_img, self.rain_img, self.bright_img,
                                    self.temp_img, self.exposed_img, self.vignette_img, self.blur_img, self.snow_img,
                                    self.pixelate_img, self.snp_img, self.gamma_img, self.rotate_img, self.flip_img, self.blend_img)
                elif rocalMode == 4:
                    for i in range(15):
                        self.copy_img = fn.copy(self.input)
                    # set outputs for this mode
                    self.pipe.set_outputs(self.copy_img, self.copy_img, self.copy_img, self.copy_img, self.copy_img,
                                    self.copy_img, self.copy_img, self.copy_img, self.copy_img, self.copy_img,
                                    self.copy_img, self.copy_img, self.copy_img, self.copy_img, self.copy_img, 
                                    self.copy_img)
                elif rocalMode == 5:
                    for i in range(15):
                        self.nop_img = fn.nop(self.input, True)
                    # set outputs for this mode
                    self.pipe.set_outputs(self.nop_img, self.nop_img, self.nop_img, self.nop_img, self.nop_img,
                                    self.nop_img, self.nop_img, self.nop_img, self.nop_img, self.nop_img,
                                    self.nop_img, self.nop_img, self.nop_img, self.nop_img, self.nop_img,
                                    self.nop_img)
            elif model_batch_size == 64:
                if rocalMode == 1:
                    self.rot150_img = fn.rotate(self.input, angle=self.degree_param_150)
                    self.flip_img = fn.flip(self.input, flip=self.flip_param)
                    self.rot45_img = fn.rotate(self.input, angle=self.degree_param_45)

                    self.setof16_mode1(self.input)
                    self.setof16_mode1(self.rot45_img)
                    self.setof16_mode1(self.flip_img)
                    self.setof16_mode1(self.rot150_img)
                    
                elif rocalMode == 2:
                    #self.warpAffine2_img = self.warpAffine(self.input, False, [[1.5,0],[0,1],[None,None]])
                    self.warpAffine1_img = fn.warp_affine(self.input, matrix=self.affine_matrix_1_param) #squeeze
                    self.fishEye_img = fn.fish_eye(self.input)
                    self.lensCorrection_img = fn.lens_correction(self.input, strength = self.strength_param, zoom = self.zoom_param)

                    self.setof16_mode1(self.input)
                    self.setof16_mode1(self.warpAffine1_img)
                    self.setof16_mode1(self.fishEye_img)
                    self.setof16_mode1(self.lensCorrection_img)

                elif rocalMode == 3:
                    self.colorTemp1_img = fn.color_temp(self.input, adjustment_value=self.adjustment_param_10)
                    self.colorTemp2_img = fn.color_temp(self.input, adjustment_value=self.adjustment_param_20)
                    self.warpAffine2_img = fn.warp_affine(self.input, matrix=self.affine_matrix_2_param) #stretch

                    self.setof16_mode1(self.input)
                    self.setof16_mode1(self.colorTemp1_img)
                    self.setof16_mode1(self.colorTemp2_img)
                    self.setof16_mode1(self.warpAffine2_img)
                elif rocalMode == 4:
                    for i in range(63):
                        self.copy_img = fn.copy(self.input)
                    # set outputs for this mode
                    self.pipe.set_outputs(self.copy_img, self.copy_img, self.copy_img, self.copy_img, self.copy_img,
                                    self.copy_img, self.copy_img, self.copy_img, self.copy_img, self.copy_img,
                                    self.copy_img, self.copy_img, self.copy_img, self.copy_img, self.copy_img, 
                                    self.copy_img, self.copy_img, self.copy_img, self.copy_img, self.copy_img, self.copy_img,
                                    self.copy_img, self.copy_img, self.copy_img, self.copy_img, self.copy_img,
                                    self.copy_img, self.copy_img, self.copy_img, self.copy_img, self.copy_img, 
                                    self.copy_img, self.copy_img, self.copy_img, self.copy_img, self.copy_img, self.copy_img,
                                    self.copy_img, self.copy_img, self.copy_img, self.copy_img, self.copy_img,
                                    self.copy_img, self.copy_img, self.copy_img, self.copy_img, self.copy_img, 
                                    self.copy_img, self.copy_img, self.copy_img, self.copy_img, self.copy_img, self.copy_img,
                                    self.copy_img, self.copy_img, self.copy_img, self.copy_img, self.copy_img,
                                    self.copy_img, self.copy_img, self.copy_img, self.copy_img, self.copy_img, 
                                    self.copy_img)
                elif rocalMode == 5:	
                    for i in range(63):
                        self.nop_img = fn.nop(self.input, True)
                    # set outputs for this mode
                    self.pipe.set_outputs(self.nop_img, self.nop_img, self.nop_img, self.nop_img, self.nop_img,
                                    self.nop_img, self.nop_img, self.nop_img, self.nop_img, self.nop_img,
                                    self.nop_img, self.nop_img, self.nop_img, self.nop_img, self.nop_img,
                                    self.nop_img, self.nop_img, self.nop_img, self.nop_img, self.nop_img, self.nop_img,
                                    self.nop_img, self.nop_img, self.nop_img, self.nop_img, self.nop_img,
                                    self.nop_img, self.nop_img, self.nop_img, self.nop_img, self.nop_img,
                                    self.nop_img, self.nop_img, self.nop_img, self.nop_img, self.nop_img, self.nop_img,
                                    self.nop_img, self.nop_img, self.nop_img, self.nop_img, self.nop_img,
                                    self.nop_img, self.nop_img, self.nop_img, self.nop_img, self.nop_img,
                                    self.nop_img, self.nop_img, self.nop_img, self.nop_img, self.nop_img, self.nop_img,
                                    self.nop_img, self.nop_img, self.nop_img, self.nop_img, self.nop_img,
                                    self.nop_img, self.nop_img, self.nop_img, self.nop_img, self.nop_img,
                                    self.nop_img)
        #rocal iterator
        self.pipe.build()
        self.tensor_format =tensor_layout
        self.multiplier = multiplier
        self.offset = offset
        self.reverse_channels = False
        self.tensor_dtype = tensor_dtype
        self.w = self.pipe.getOutputWidth()
        self.h = self.pipe.getOutputHeight()
        self.b = self.pipe._batch_size
        self.n = self.pipe.getOutputImageCount()
        color_format = self.pipe.getOutputColorFormat()
        self.p = (1 if color_format is types.GRAY else 3)
        height = self.h*self.n
        self.out_image = np.zeros((height, self.w, self.p), dtype = "uint8")
        self.out_tensor = np.zeros(( self.b*self.n, self.p, int(self.h/self.b), self.w,), dtype = "float32")

    def get_input_name(self):
        self.img_names_length = np.empty(self.rocal_batch_size, dtype="int32")
        self.img_names_size = self.pipe.GetImageNameLen(self.img_names_length)
        # Images names of a batch
        self.Img_name = self.pipe.GetImageName(self.img_names_size)
        return self.Img_name.decode()

    def process_validation(self, validation_list):
        for i in range(len(validation_list)):
            name, groundTruthIndex = validation_list[i].split(' ')
            self.validation_dict[name] = groundTruthIndex

    def get_ground_truth(self):
        return self.validation_dict[self.get_input_name()]

    def setof16_mode1(self, input_image):
        self.warped = fn.warp_affine(input_image, matrix=self.affine_matrix_param)
        self.contrast_img = fn.contrast(input_image, min_contrast=self.min_param, max_contrast=self.max_param)
        self.rain_img = fn.rain(input_image, rain=self.rain_param, rain_width = self.rain_width_param, rain_height = self.rain_height_param, rain_transparency =self.rain_transparency_param)
        self.bright_img = fn.brightness_fixed(input_image, alpha=self.alpha_param, beta= self.beta_param)
        self.temp_img = fn.color_temp(input_image, adjustment_value=self.adjustment_param)
        self.exposed_img = fn.exposure(input_image, exposure=self.shift_param)
        self.vignette_img = fn.vignette(input_image, vignette = self.vignette_param)
        self.blur_img = fn.blur(input_image, blur = self.blur_param)
        self.snow_img = fn.snow(input_image, snow=self.snow_param)
        self.pixelate_img = fn.pixelate(input_image)
        self.snp_img = fn.snp_noise(input_image, snpNoise=self.sdev_param)
        self.gamma_img = fn.gamma_correction(input_image, gamma=self.gamma_shift_param)
        self.rotate_img = fn.rotate(input_image, angle=self.degree_param)
        self.flip_img = fn.flip(input_image, flip=self.flip_param)
        self.blend_img = fn.blend(input_image, self.contrast_img)

        # set outputs for this mode
        self.pipe.set_outputs(input_image, self.warped, self.contrast_img, self.rain_img, self.bright_img,
                        self.temp_img, self.exposed_img, self.vignette_img, self.blur_img, self.snow_img,
                        self.pixelate_img, self.snp_img, self.gamma_img, self.rotate_img, self.flip_img, self.blend_img)

    def updateAugmentationParameter(self, augmentation):
        #values for contrast
        self.aug_strength = augmentation
        min = int(augmentation*100)
        max = 150 + int((1-augmentation)*100)
        self.pipe.update_int_param(min, self.min_param)
        self.pipe.update_int_param(max, self.max_param)

        #values for brightness
        alpha = augmentation*1.95
        self.pipe.update_float_param(alpha, self.alpha_param)

        #values for ColorTemperature
        adjustment = (augmentation*99) if ((int(augmentation*100)) % 2 == 0) else (-1*augmentation*99)
        adjustment = int(adjustment)
        self.pipe.update_int_param(adjustment, self.adjustment_param)

        #values for exposure
        shift = augmentation*0.95
        self.pipe.update_float_param(shift, self.shift_param)

        #values for SnPNoise
        sdev = augmentation*0.7
        self.pipe.update_float_param(sdev, self.sdev_param)

        #values for gamma
        gamma_shift = augmentation*5.0
        self.pipe.update_float_param(gamma_shift, self.gamma_shift_param)

    def renew_parameters(self):
        curr_degree = self.pipe.get_float_value(self.degree_param)
        #values for rotation change
        degree = self.aug_strength * 100
        self.pipe.update_float_param(curr_degree+degree, self.degree_param)

    def start_iterator(self):
        self.rocalResetLoaders()
        
    def get_next_augmentation(self, imageIterator):
        if self.pipe.isEmpty() == 1:
            return -1
            #raise StopIteration
        self.renew_parameters()
        self.out_image = imageIterator.next()
        if(types.NCHW == self.tensor_layout):
            self.pipe.copyToTensorNCHW(self.out_tensor, self.multiplier, self.offset, self.reverse_channels, int(self.tensor_dtype))
        else:
            self.pipe.copyToTensorNHWC(self.out_tensor, self.multiplier, self.offset, self.reverse_channels, int(self.tensor_dtype))

        return self.out_image, self.out_tensor

    def get_rocal_list(self, rocalMode, model_batch_size):
        if model_batch_size == 16:
            if rocalMode == 1:
                self.rocal_list = rocalList_mode1_16
            elif rocalMode == 2:
                self.rocal_list = rocalList_mode2_16
            elif rocalMode == 3:
                self.rocal_list = rocalList_mode3_16
            elif rocalMode == 4:
                self.rocal_list = rocalList_mode4_16
            elif rocalMode == 5:
                self.rocal_list = rocalList_mode5_16
        elif model_batch_size == 64:
            if rocalMode == 1:
                self.rocal_list = rocalList_mode1_64
            elif rocalMode == 2:
                self.rocal_list = rocalList_mode2_64
            elif rocalMode == 3:
                self.rocal_list = rocalList_mode3_64
            elif rocalMode == 4:
                self.rocal_list = rocalList_mode4_64
            elif rocalMode == 5:
                self.rocal_list = rocalList_mode5_64
                
        return self.rocal_list
