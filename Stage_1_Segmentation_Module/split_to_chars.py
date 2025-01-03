import os
import cv2
import numpy as np


class SplitToChars:
    """
    A class used to segment an image of text into individual characters (and optional spaces).

    Steps performed:
      1) Read a grayscale image and threshold it (binarize).
      2) Invert it so text appears white on black.
      3) Use horizontal projections to detect rows (line boundaries).
      4) Within each row, detect words by contour analysis (dilation).
      5) Within each word, detect individual character contours.
      6) Crop, center, and optionally rescale each character before saving.
      7) Optionally insert a placeholder for spaces between words.
    """
    def __init__(self, padding=10, proportional_padding=False):
        """
        Initialize the ImageProcessor with padding options.

        Args:
            padding (int): Fixed padding size for cropping or bounding boxes.
            proportional_padding (bool):
                If True, use proportional padding (currently not used in code,
                but kept as a design parameter if needed).
        """
        self.padding = padding                              # Store the fixed padding size
        self.proportional_padding = proportional_padding    # Whether we use proportional padding

    def process_image(self, image_path, output_folder):
        """
        Segment an image into characters (and spaces between words), then save them.

        Args:
            image_path (str):
                The full path to the input image file.
            output_folder (str):
                A directory where the segmented character and space images will be saved.

        Returns:
            list:
                A list of file paths pointing to the saved character and space images.
        """
        # -----------------------------------------------------------
        # 1. Read the image in grayscale
        # -----------------------------------------------------------
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)        # Load grayscale image
        if image is None:
            print(f"Error reading image: {image_path}")
            return []

        # -----------------------------------------------------------
        # 2. Threshold (binarize) and invert so text is white on black
        #
        #    - THRESH_BINARY_INV inverts the result (white text on black).
        #    - Otsu's threshold (cv2.THRESH_OTSU) automatically finds a suitable threshold.
        # -----------------------------------------------------------
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # -----------------------------------------------------------
        # 3. Use horizontal projection to find row boundaries
        #
        #    - Summation of pixel values across each row (axis=1).
        #    - The _find_gaps() method uses these sums to detect where rows start/end.
        # -----------------------------------------------------------
        projection = np.sum(thresh, axis=1)     # Summation over columns => row "signal"
        row_gaps = self._find_gaps(projection)  # Use _find_gaps method to find line boundaries

        # Ensure output directory exists
        os.makedirs(output_folder, exist_ok=True)

        # We'll store the resulting image paths and track a segment numbering
        segment_images = []         # Will store file paths of resulting images
        segment_count = 1           # Counter to name each segment (letter or space)

        # -----------------------------------------------------------
        # 4. For each consecutive pair of gaps, we have one row region.
        #    row_gaps[i], row_gaps[i+1] define the vertical boundaries of that row.
        # -----------------------------------------------------------
        for i in range(len(row_gaps) - 1):
            y_start = row_gaps[i]               # Start row index
            y_end = row_gaps[i + 1]             # End row index

            # Extract the row region from the thresholded image
            row = thresh[y_start:y_end, :]  # binarized row region

            # -------------------------------------------------------
            # 4.1: Detect words in the row by applying dilation horizontally
            #      to group letters in each word together.
            # -------------------------------------------------------
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
            dilated_row = cv2.dilate(row, kernel, iterations=1)

            # Find word contours
            word_contours, _ = cv2.findContours(
                dilated_row, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            # Sort by x-position so we process words left-to-right
            word_contours = sorted(word_contours, key=lambda c: cv2.boundingRect(c)[0])

            # Process each word in this row
            for w_idx, wc in enumerate(word_contours):
                wx, wy, ww, wh = cv2.boundingRect(wc)

                # Crop the word from the ORIGINAL row
                word_img = row[wy : wy + wh, wx : wx + ww]

                # ---------------------------------------------------
                # 4.2: Process characters within the word
                #      This adds the segmented character images to 'segment_images'
                #      and returns the updated segment_count.
                # ---------------------------------------------------
                segment_count = self._process_word(
                    word_img, segment_images, segment_count, output_folder
                )

                # Insert exactly one space image between words (but not after the last word in a row)
                if w_idx < len(word_contours) - 1:
                    space_image = np.full((28, 28), 255, dtype=np.uint8)
                    space_path = os.path.join(output_folder, f"{segment_count}.png")
                    cv2.imwrite(space_path, space_image)
                    segment_images.append(space_path)
                    segment_count += 1

        print(f"Processed {len(segment_images)} segments and saved to '{output_folder}'."
        )
        return segment_images

    def _process_word(self, word_img, segment_images, segment_count, output_folder):
        """
        Process a single word image by detecting individual character contours
        and saving them to separate image files.

        Args:
            word_img (ndarray):
                The cropped (binarized) image of the word region.
            segment_images (list):
                A list to which we append file paths of the saved character images.
            segment_count (int):
                The integer index used to name the next segment file.
            output_folder (str):
                A directory where the segmented letter images are saved.

        Returns:
            int:
                The updated segment_count after processing this word.
        """
        # Find contours for individual letters in the word
        char_contours, _ = cv2.findContours(
            word_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Sort letters by their x-position, so left-to-right
        char_contours = sorted(char_contours, key=lambda c: cv2.boundingRect(c)[0])

        # Loop through each letter contour
        for idx, contour in enumerate(char_contours):
            x, y, w, h = cv2.boundingRect(contour)

            # Filter out noise or extremely small contours
            if w < 2 or h < 2:
                continue

            # Extract the character region from the word
            char_crop = word_img[y : y + h, x : x + w]

            # Convert black-on-white to white-on-black by inverting
            char_inv = cv2.bitwise_not(char_crop)

            # -------------------------------
            # Decide on a scale factor:
            # If the character bounding box is very small (<=5 in width or height),
            # we treat it as punctuation and scale it less (e.g., 0.8).
            # Otherwise, treat as normal letter and scale more (e.g., 1.5).
            # -------------------------------
            if w <= 5 and h <= 5:
                # This is presumably punctuation => smaller scale
                scale_factor = 0.8
            else:
                # Normal letter scale
                scale_factor = 1.5

            # Center and resize the character to a 28x28 canvas
            centered_char = self._center_image(
                char_inv, target_size=(28, 28), scale_factor=scale_factor
            )

            # Save the character
            char_path = os.path.join(output_folder, f"{segment_count}.png")
            cv2.imwrite(char_path, centered_char)
            segment_images.append(char_path)
            segment_count += 1

        return segment_count

    def _find_gaps(self, projection):
        """
        Given a horizontal projection of the image, locate the indices
        where text lines start and end.

        Args:
            projection (ndarray):
                The 1D array representing the sum of pixel values in each row.

        Returns:
            list:
                A list of row indices marking boundaries. For example:
                [start_line1, end_line1, start_line2, end_line2, ...].
        """
        gaps = []
        in_gap = True        # We'll track if we're currently in a gap (no text) or not

        for i, value in enumerate(projection):
            if value > 0 and in_gap:
                # We encountered text after a gap -> start of a row
                gaps.append(i)
                in_gap = False
            elif value == 0 and not in_gap:
                # We encountered a zero after text -> end of a row
                gaps.append(i)
                in_gap = True

        # If the last row extends to the end of the image, close it here
        if not in_gap:
            gaps.append(len(projection))

        return gaps

    def _center_image(self, segment, target_size=(28, 28), scale_factor=1.3):
        """
        Scale and center a character (segment) within a specified canvas (28x28 by default).

        Args:
            segment (ndarray):
                The white-on-black character image to be centered.
            target_size (tuple):
                Desired (height, width) for the output canvas, e.g. (28, 28).
            scale_factor (float):
                Factor that determines how much we scale the bounding box before
                placing it in the 28x28 canvas.

        Returns:
            ndarray:
                The new 28x28 (or specified size) image with the character centered.
        """
        canvas = np.full(target_size, 255, dtype=np.uint8)  # Create a blank white canvas
        h, w = segment.shape                                        # Original height and width of the character

        # Compute a scale such that the character doesn't exceed
        scale = min(
            target_size[0] / (h * scale_factor),
            target_size[1] / (w * scale_factor)
        )

        if scale > 1:
            scale = 1

        new_h = int(h * scale)
        new_w = int(w * scale)

        #  Resize the segment with OpenCV interpolation
        resized_segment = cv2.resize(segment, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Compute offsets so the character is centered in a 28x28 area
        y_offset = (target_size[0] - new_h) // 2
        x_offset = (target_size[1] - new_w) // 2

        # Place it in the center
        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized_segment

        return canvas