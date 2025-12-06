import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

PERCENT_OF_AVERAGE_TOP5_FOR_FINAL_FILTER_THRESHOLD = 0.3  # 30% of average of top 5 counts


class FilteringEngine:
    def __init__(self):
        self.percent_for_final_threshold = PERCENT_OF_AVERAGE_TOP5_FOR_FINAL_FILTER_THRESHOLD
    
    def compress_consecutive_detections(self, detection_history):
        if not detection_history:
            return []
        
        compressed = []
        current_letter = None
        current_count = 0
        
        for frame_num, letter, confidence in detection_history:
            if letter != current_letter:
                if current_letter is not None:
                    compressed.append((current_letter, current_count))
                current_letter = letter
                current_count = 1
            else:
                current_count += 1
        
        if current_letter is not None:
            compressed.append((current_letter, current_count))
        
        return compressed
    
    def merge_consecutive_same(self, detections):
        if not detections:
            return []
        
        merged = []
        current_item = None
        current_total_count = 0
        
        for item, count in detections:
            if item != current_item:
                if current_item is not None:
                    merged.append((current_item, current_total_count))
                current_item = item
                current_total_count = count
            else:
                current_total_count += count
        
        if current_item is not None:
            merged.append((current_item, current_total_count))
        
        return merged
    
    def calculate_average_of_top5(self, compressed_detections):
        """Calculate the average of the top 5 count values"""
        if not compressed_detections:
            return 0
        
        # Extract all counts
        counts = [count for _, count in compressed_detections]
        
        # Sort counts in descending order
        counts.sort(reverse=True)
        
        # Take the top 5 or all available if less than 5
        top_counts = counts[:5]
        
        # Calculate average
        average_top5 = sum(top_counts) / len(top_counts)
        
        logger.info(f"Top {len(top_counts)} counts: {top_counts}")
        logger.info(f"Average of top {len(top_counts)}: {average_top5:.2f}")
        
        return average_top5
    
    def calculate_dynamic_threshold(self, compressed_detections):
        """Calculate dynamic threshold as a percentage of the average of top 5 counts"""
        if not compressed_detections:
            return 3  # Minimum value (2 + 1)
        
        # Calculate average of top 5 counts
        average_top5 = self.calculate_average_of_top5(compressed_detections)
        
        if average_top5 > 0:
            # Calculate percentage of the average top 5
            dynamic_threshold = int(average_top5 * self.percent_for_final_threshold)
            
            # Ensure threshold is at least 3 (2 + 1)
            dynamic_threshold = max(3, dynamic_threshold)
            
            logger.info(f"Dynamic threshold calculated: {dynamic_threshold} ({self.percent_for_final_threshold*100}% of average top 5: {average_top5:.2f})")
            return dynamic_threshold
        
        return 3  # Minimum value (2 + 1)
    
    def apply_recursive_filter(self, compressed_detections):
        if not compressed_detections:
            return []
        
        # Calculate dynamic threshold based on average of top 5 counts
        dynamic_threshold = self.calculate_dynamic_threshold(compressed_detections)
        
        current_threshold = 2  # Hardcoded initial threshold
        current_data = compressed_detections.copy()
        
        while current_threshold <= dynamic_threshold:
            filtered_data = []
            for item, count in current_data:
                if count >= current_threshold:
                    filtered_data.append((item, count))
            
            grouped_data = self.merge_consecutive_same(filtered_data)
            current_data = grouped_data
            current_threshold += 1  # Hardcoded increment of 1
        
        return current_data