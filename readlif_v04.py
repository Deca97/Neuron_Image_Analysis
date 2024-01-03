import tkinter as tk
from tkinter import filedialog as fd
import read_lif
import cv2
import numpy as np
from scipy import stats
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt
from tkinter import ttk



class ImageProcessorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Processor")
        self.filename = None
        self.current_index = 0
        self.scale = 30.864
        self.master.geometry("400x250")
        self.master.resizable(False, False)  # Disable resizing in both dimensions





        self.frame = tk.Frame(master)
        self.frame.pack(expand=True, fill='both')

        # Create labels and entry widgets for parameters
        self.radius_label = tk.Label(self.frame, text="Radius for background removal:")
        self.radius_label.grid(row=0, column=0, padx=10, pady=10)
        self.radius_entry = tk.Entry(self.frame)
        self.radius_entry.grid(row=0, column=1, padx=10, pady=10)

        self.tr_label = tk.Label(self.frame, text="Image processing threshold [0-255]:")
        self.tr_label.grid(row=1, column=0, padx=10, pady=10)
        self.tr_entry = tk.Entry(self.frame)
        self.tr_entry.grid(row=1, column=1, padx=10, pady=10)
        

        self.threshold_label = tk.Label(self.frame, text="Overlapping sensitivity threshold [0.0-1.0]:")
        self.threshold_label.grid(row=2, column=0, padx=10, pady=10)
        self.threshold_entry = tk.Entry(self.frame)
        self.threshold_entry.grid(row=2, column=1, padx=10, pady=10)

        # Create a button to open the file dialog
        self.open_button = tk.Button(self.frame, text="Open LIF File", command=self.open_file)
        self.open_button.grid(row=3, column=0, columnspan=2, pady=10)

        # Create a button to execute image processing
        self.process_button = tk.Button(self.frame, text="Process Image", command=self.process_image)
        self.process_button.grid(row=4, column=0, columnspan=2, pady=10)

    def open_file(self):
        self.filename = fd.askopenfilename(title='Open LIF file', filetypes=[('LIF files', '.lif')])

    def process_image(self):
        if self.filename:
            try:
                radius = int(self.radius_entry.get())
                tr = int(self.tr_entry.get())
                threshold = float(self.threshold_entry.get())

                file = read_lif.Reader(self.filename)

                series = file.getSeries()





                name_info = {}

                for k in range(3, len(series)):

                    chosen = series[k]  
                    frame = chosen.getFrame(channel = 1)

                    images = []
                    
                    name = chosen.getName()
                    name = name[0:4]

                    if name not in name_info:
                        name_info[name] = {
                        'num_dots_verde': [],
                        'area_mean_verde': [],
                        'num_dots_rosso': [],
                        'area_mean_rosso': [],
                        'overlapping_dots': []
                        }     

                    for n in range(len(frame)):
                    
                        images.append(frame[n])



                    # Images to keep
                    selected_images = set()

                    self.current_index = 0

                    # Display the current image
                    def display_current_image():
                        ax.clear()
                        ax.imshow(images[self.current_index])  
                        plt.title(f"Name {chosen.getName()} \nImage {self.current_index + 1}/{len(images)}\nPress any key to KEEP, Click to DISCARD")

                        plt.draw()

                    # Keep the current image
                    def keep_image(event):
                        selected_images.add(self.current_index)
                        next_image(event)

                    # Discard the current image
                    def discard_image(event):
                        next_image(event)



                    # Next image
                    def next_image(event):

                        self.current_index = (self.current_index + 1) % len(images)

                        display_current_image()

                    # Set up Matplotlib figure and axes
                    fig, ax = plt.subplots()
                    fig.canvas.mpl_connect('key_press_event', keep_image)  # Press any key to keep the image
                    fig.canvas.mpl_connect('button_press_event', discard_image)  # Click to discard the image

                    # Display the first image
                    display_current_image()

                    # Show the plot
                    plt.show()

                    # After looking at all the images and making selections, you can access the selected_images set
                    print(f"Name: {chosen.getName()}: Selected Images:", selected_images)

                    for ch in range(2):

                        frame = chosen.getFrame(channel = ch) #0 = Green, 1 = Red

                        img = [frame[i] for i in selected_images]



                        
                        img_max = np.max(img, axis = 0)

                        if ch == 0:

                            # Selected region coordinates
                            selected_region = [None]

                            # Display the projected image with an interactive cropping interface
                            def display_projected_image():
                                fig, ax = plt.subplots()
                                ax.imshow(img_max)
                                plt.title("Manually Crop Projected Image\nClick and drag to draw a rectangle")

                                def onselect(eclick, erelease):
                                    x1, y1 = eclick.xdata, eclick.ydata
                                    x2, y2 = erelease.xdata, erelease.ydata
                                    selected_region[0] = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                                    plt.close(fig)

                                rectprops = dict(facecolor='none', edgecolor='red', linewidth=2, linestyle='dashed')
                                span_selector = RectangleSelector(ax, onselect, drawtype='box', rectprops=rectprops, spancoords='data')

                                plt.show()

                            # Display the projected image and allow the user to manually crop it
                            display_projected_image()

                            # Crop the projected image based on the selected region
                            if selected_region[0]:
                                x1, y1, x2, y2 = selected_region[0]
                                cropped_projected_image = img_max[int(y1):int(y2), int(x1):int(x2)]


                                # Print the dimensions of the cropped projected image
                                cropped_dimensions = np.shape(cropped_projected_image)
                                print("Dimensions of Cropped Projected Image:", cropped_dimensions)
                            else:
                                print("No region selected.")

                            region_for_red_ch = selected_region[0]

                        if ch == 1:
                            if region_for_red_ch:
                                x1, y1, x2, y2 = region_for_red_ch
                                cropped_projected_image = img_max[int(y1):int(y2), int(x1):int(x2)]



                                # Print the dimensions of the cropped projected image
                                cropped_dimensions = np.shape(cropped_projected_image)
                                print("Dimensions of Cropped Projected Image:", cropped_dimensions)

                        
                        scale_factor = 0.18*0.18*cropped_dimensions[0]*cropped_dimensions[1]/1000
                        



                        # Apply a morphological opening with a circular structuring element
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
                        opened = cv2.morphologyEx(cropped_projected_image, cv2.MORPH_OPEN, kernel)

                        # Calculate the difference between the original image and the opened image
                        corrected = cv2.subtract(cropped_projected_image, opened)

                        # Apply thresholding 
                        _, thresh = cv2.threshold(corrected, tr, 255, cv2.THRESH_BINARY)

                        # Find contours in the binary image
                        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


                        area_dots = []
                        filtered_contours = []

                        for contour in contours:
                        # Calculate the area of each contour
                            area = cv2.contourArea(contour)/self.scale
                            if area < 3 and area > 0.2: 
                                area_dots.append(area)
                                filtered_contours.append(contour)

                        
                        if ch == 0:
                            canale = 'Verde'
                            filtered_contours_verde = filtered_contours.copy()
                        if ch == 1:
                            canale = 'Rosso'
                            filtered_contours_rosso = filtered_contours.copy()

                        # Draw contours on the original image 
                        img_contours = cropped_projected_image.copy()
                        cv2.drawContours(img_contours, filtered_contours, -1, (255, 255, 255), 10)

                        # Display the original, thresholded, and contoured images
                        plt.subplot(221), plt.imshow(img_max), plt.title('Original Image')
                        plt.subplot(222), plt.imshow(cropped_projected_image), plt.title('Cropped Image')
                        plt.subplot(223), plt.imshow(thresh), plt.title('Thresholded Image')
                        plt.subplot(224), plt.imshow(img_contours), plt.title('Contoured Image')
                        plt.show()
                                
                            
                        # Count the number of contours (dots)/um^2 and mean area
                        num_dots = len(filtered_contours)/scale_factor
                        area_mean = np.mean(area_dots)


                        


                        if ch == 0:
                            name_info[name]['num_dots_verde'].append(num_dots)
                            name_info[name]['area_mean_verde'].append(area_mean)
                        if ch == 1:
                            name_info[name]['num_dots_rosso'].append(num_dots)
                            name_info[name]['area_mean_rosso'].append(area_mean)

                        


                            
                        print('\n')
                        print(name)
                        print('Channel: ', canale)
                        print('Number of dots/1000 um^2: ', num_dots)
                        print('Average dots area (um^2): ', area_mean)


                    overlapping_contours = []
                    match_score = []


                    # Iterate through contours in the first image
                    for contour1 in filtered_contours_verde:
                        # Iterate through contours in the second image
                        for contour2 in filtered_contours_rosso:

                            # Compare contours using matchShapes
                            x1, y1, w1, h1 = cv2.boundingRect(contour1)
                            x2, y2, w2, h2 = cv2.boundingRect(contour2)

                            intersection_area = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

                            area1 = cv2.contourArea(contour1)
                            area2 = cv2.contourArea(contour2)

                            # Calculate the overlap ratio
                            overlap_ratio = intersection_area / min(area1, area2)

                            if overlap_ratio > threshold: 
                                overlapping_contours.append(overlap_ratio)
            

                    overlapping_dots = len(overlapping_contours)/scale_factor * 1000

                    print('\nNumber of common points between chennels/1000 um^2: ', overlapping_dots)



                    name_info[name]['overlapping_dots'].append(overlapping_dots)


                # Calculate the mean information within each name
                for name, info in name_info.items():
                    info['mean_num_dots_verde'] = np.mean(info['num_dots_verde'])
                    info['mean_area_mean_verde'] = np.mean(info['area_mean_verde'])
                    info['mean_num_dots_rosso'] = np.mean(info['num_dots_rosso'])
                    info['mean_area_mean_rosso'] = np.mean(info['area_mean_rosso'])
                    info['mean_overlapping'] = np.mean(info['overlapping_dots'])

                # Split subjects into two groups based on the first two letters of their names
                KO_subjects = {subject: info for subject, info in name_info.items() if subject[:2] == 'KO'}
                WT_subjects = {subject: info for subject, info in name_info.items() if subject[:2] == 'WT'}

                # Convert the dictionary to a DataFrame
                df_KO = pd.DataFrame.from_dict(KO_subjects, orient='index')
                df_WT = pd.DataFrame.from_dict(WT_subjects, orient='index')


                # Save the DataFrame to an Excel file
                KO_excel_file_path = 'KO.xlsx'
                df_KO.to_excel(KO_excel_file_path, index_label='Key')
                WT_excel_file_path = 'WT.xlsx'
                df_WT.to_excel(WT_excel_file_path, index_label='Key')



                dots_KO_verde = [info['mean_num_dots_verde'] for info in KO_subjects.values()]
                dots_KO_rosso = [info['mean_num_dots_rosso'] for info in KO_subjects.values()]
                dots_WT_verde = [info['mean_num_dots_verde'] for info in WT_subjects.values()]
                dots_WT_rosso = [info['mean_num_dots_rosso'] for info in WT_subjects.values()]
                areas_KO_verde = [info['mean_area_mean_verde'] for info in KO_subjects.values()]
                areas_KO_rosso = [info['mean_area_mean_rosso'] for info in KO_subjects.values()]
                areas_WT_verde = [info['mean_area_mean_verde'] for info in WT_subjects.values()]
                areas_WT_rosso = [info['mean_area_mean_rosso'] for info in WT_subjects.values()]
                overlap_KO = [info['mean_overlapping'] for info in KO_subjects.values()]
                overlap_WT = [info['mean_overlapping'] for info in WT_subjects.values()]



                # Perform a two-sample t-test
                dots_t_statistic_verde, dots_p_value_verde = stats.ttest_ind(dots_KO_verde, dots_WT_verde)
                areas_t_statistic_verde, areas_p_value_verde = stats.ttest_ind(areas_KO_verde, areas_WT_verde)
                dots_t_statistic_rosso, dots_p_value_rosso = stats.ttest_ind(dots_KO_rosso, dots_WT_rosso)
                areas_t_statistic_rosso, areas_p_value_rosso = stats.ttest_ind(areas_KO_rosso, areas_WT_rosso)
                overlap_t_statistic, overlap_p_value = stats.ttest_ind(overlap_KO, overlap_WT)


                # Output the results
                print('\n')
                print(f"Dots P-value canale verde: {dots_p_value_verde}")
                print(f"Areas P-value canale verde: {areas_p_value_verde}")
                print('\n')
                print(f"Dots P-value canale rosso: {dots_p_value_rosso}")
                print(f"Areas P-value canale rosso: {areas_p_value_rosso}")
                print('\n')
                print(f"Overlap P-value: {overlap_p_value}")

                plt.figure(1)

                plt.subplot(221)
                plt.scatter(np.ones(len(dots_WT_verde)), dots_WT_verde, label = 'WT')
                plt.scatter(2*np.ones(len(dots_KO_verde)), dots_KO_verde, label = 'KO')
                plt.plot([0.6, 1.4], [np.mean(dots_WT_verde), np.mean(dots_WT_verde)], label = 'WT mean', marker = '_')
                plt.plot([1.6, 2.4], [np.mean(dots_KO_verde), np.mean(dots_KO_verde)], label = 'KO mean', marker = '_')
                plt.title(f"Dots P-value canale verde: {round(dots_p_value_verde, 3)}")
                plt.xticks([1, 2], ['WT', 'KO'])
                plt.ylabel('Number of dots/1000 um^2')
                plt.legend()
                plt.xlim([0, 3])



                plt.subplot(222)
                plt.scatter(np.ones(len(areas_WT_verde)), areas_WT_verde, label = 'WT')
                plt.scatter(2*np.ones(len(areas_KO_verde)), areas_KO_verde, label = 'KO')
                plt.plot([0.6, 1.4], [np.mean(areas_WT_verde), np.mean(areas_WT_verde)], label = 'WT mean', marker = '_')
                plt.plot([1.6, 2.4], [np.mean(areas_KO_verde), np.mean(areas_KO_verde)], label = 'KO mean', marker = '_')
                plt.title(f"Areas P-value canale verde: {round(areas_p_value_verde, 3)}")
                plt.xticks([1, 2], ['WT', 'KO'])
                plt.ylabel('Average dots area (um^2)')
                plt.legend()
                plt.xlim([0, 3])


                plt.subplot(223)
                plt.scatter(np.ones(len(dots_WT_rosso)), dots_WT_rosso, label = 'WT')
                plt.scatter(2*np.ones(len(dots_KO_rosso)), dots_KO_rosso, label = 'KO')
                plt.plot([0.6, 1.4], [np.mean(dots_WT_rosso), np.mean(dots_WT_rosso)], label = 'WT mean', marker = '_')
                plt.plot([1.6, 2.4], [np.mean(dots_KO_rosso), np.mean(dots_KO_rosso)], label = 'KO mean', marker = '_')
                plt.title(f"Dots P-value canale rosso: {round(dots_p_value_rosso, 3)}")
                plt.xticks([1, 2], ['WT', 'KO'])
                plt.ylabel('Number of dots/1000 um^2')
                plt.legend()
                plt.xlim([0, 3])



                plt.subplot(224)
                plt.scatter(np.ones(len(areas_WT_rosso)), areas_WT_rosso, label = 'WT')
                plt.scatter(2*np.ones(len(areas_KO_rosso)), areas_KO_rosso, label = 'KO')
                plt.plot([0.6, 1.4], [np.mean(areas_WT_rosso), np.mean(areas_WT_rosso)], label = 'WT mean', marker = '_')
                plt.plot([1.6, 2.4], [np.mean(areas_KO_rosso), np.mean(areas_KO_rosso)], label = 'KO mean', marker = '_')
                plt.title(f"Areas P-value canale rosso: {round(areas_p_value_rosso, 3)}")
                plt.xticks([1, 2], ['WT', 'KO'])
                plt.ylabel('Average dots area (um^2)')
                plt.legend()
                plt.xlim([0, 3])

                plt.show()

                plt.figure(2)
                plt.scatter(np.ones(len(overlap_WT)), overlap_WT, label = 'WT')
                plt.scatter(2*np.ones(len(overlap_KO)), overlap_KO, label = 'KO')
                plt.plot([0.6, 1.4], [np.mean(overlap_WT), np.mean(overlap_WT)], label = 'WT mean', marker = '_')
                plt.plot([1.6, 2.4], [np.mean(overlap_KO), np.mean(overlap_KO)], label = 'KO mean', marker = '_')
                plt.title(f"Overlap P-value: {round(overlap_p_value, 3)}")
                plt.xticks([1, 2], ['WT', 'KO'])
                plt.ylabel('Number of overlapping dots/1000 um^2')
                plt.legend()
                plt.xlim([0, 3])

                plt.show()





            except ValueError:
                print("Error")


def main():
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()










