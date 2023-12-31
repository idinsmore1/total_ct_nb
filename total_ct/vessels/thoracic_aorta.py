import cupy as cp
from cucim.skimage.measure import label, regionprops
from total_ct.vessels.cpr import CurvedPlanarReformation, measure_binary_diameter

class ThoracicAortaCPR(CurvedPlanarReformation):
    """Special instance of the CPR class - need a split based on the aortic arch and two different measurments"""
    def run_process(self, grid_size=80):
        """Method to run and return the outputs of find_max_diameter"""
        self.create_centerline(self.kimimaro_const)
        if self.centerline is None:
            return {}, {}
        self.create_straightened_cpr(grid_size)
        self.measure_diameters()
        measurements = self.find_max_diameter()
        return measurements
    
    def measure_diameters(self):
        "Method to handle the measurement of the ascending and descending aorta as two separate entities"
        self._split_thoracic_aorta()
        for name, centerline, cpr in zip(['asc', 'desc'], [self.asc_centerline, self.desc_centerline], [self.asc_cpr, self.desc_cpr]):
            measurements = {}
            if name == 'asc':
                iterator = range(int(len(centerline) * 0.4), centerline.shape[0])
            else:
                iterator = range(int(len(centerline) * 0.125), centerline.shape[0])
            for i in iterator:
                if centerline[i][0] >= self.arch_split:
                    roi = self._smooth_cpr_slice(cpr[i])
                    if roi is not None and roi.sum() > 0:
                        diameters = measure_binary_diameter(roi, self.vspacing[1:])
                        if diameters is not None:
                            avg_diam, major_diam, minor_diam = diameters
                            measurements[i] = {
                                'avg_diam': avg_diam,
                                'major_diam': major_diam,
                                'minor_diam': minor_diam,
                                'slice_idx': round(float(centerline[i][0]), 1)
                            }
            if len(measurements) == 0:
                measurements = None
            if name == 'asc':
                self.asc_diameters = measurements
            else:
                self.desc_diameters = measurements
                
    def find_max_diameter(self):
        """Method to find the max diameter of the ascending and descending aorta
        """
        if self.asc_diameters is not None:
            asc_max, asc_max_key = self.get_max_diameter(self.asc_diameters)
            asc_comps = self.closest_measured_diameters(self.asc_diameters, asc_max_key)
            asc_max['comp_diams'] = asc_comps
        else:
            asc_max = {}
        if self.desc_diameters is not None:
            desc_max, desc_max_key = self.get_max_diameter(self.desc_diameters)
            desc_comps = self.closest_measured_diameters(self.desc_diameters, desc_max_key)
            desc_max['comp_diams'] = desc_comps
        else:
            desc_max = {}
        return asc_max, desc_max
    
    def _split_thoracic_aorta(self):
        arch_peak = round(float(cp.argmin(self.centerline, axis=0)[0]))
        self.asc_cpr = self.cpr[:arch_peak]
        self.asc_centerline = self.centerline[:arch_peak]
        self.desc_cpr = self.cpr[arch_peak:]
        self.desc_centerline = self.centerline[arch_peak:]
        self.arch_split = self._find_arch_split()
        
    def _find_arch_split(self, min_pixel_area=300) -> int:
        """Internal method to find the z-slice that represents the end of the aortic arch
        :param min_pixel_area: the minimum area that both asc and desc aorta need to be to finalize z-slice
        :return: the split index
        """
        split = 0
        for i in range(self.vessel.shape[0]):
            labels = label(self.vessel[i])
            regions = regionprops(labels)
            if len(regions) == 2:
                reg0 = regions[0]
                reg1 = regions[1]
                if reg0.area > min_pixel_area and reg1.area > min_pixel_area:
                    split = i
                    break
        return split