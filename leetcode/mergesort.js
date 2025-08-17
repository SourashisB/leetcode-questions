    var findMedianSortedArrays = function mergeSortAndFindMedian(nums1, nums2) {
        function mergeArrays(nums1, nums2) {
            let merged = [];
            let i = 0;
            let j = 0;
        
            while (i < nums1.length && j < nums2.length) {
            if (nums1[i] < nums2[j]) {
                merged.push(nums1[i]);
                i++;
            } else {
                merged.push(nums2[j]);
                j++;
            }
            }
        
            while (i < nums1.length) {
            merged.push(nums1[i]);
            i++;
            }
        
            while (j < nums2.length) {
            merged.push(nums2[j]);
            j++;
            }
        
            return merged;
        }
        
        const mergedArray = mergeArrays(nums1, nums2);
        const medianIndex = Math.floor(mergedArray.length / 2);
    
        if (mergedArray.length % 2 === 0) {
        // even
        const median = (mergedArray[medianIndex - 1] + mergedArray[medianIndex]) / 2;
        return median;
        } else {
        // odd
        return mergedArray[medianIndex];
        }
    }
    
  