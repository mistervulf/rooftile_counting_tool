import os
import re
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from collections import Counter
from tkinter import Toplevel
from scipy.signal import find_peaks


# Globale Variablen
global cv_image, photo, canvas, defined_point,i


#################################################################### Funktionen ########################################################################
def fourier_from_list(list_points):
    """This function does the fourier-analysye
    Parameters:
        - list with values
    Output:
        - period duration
        number of tiles
    """
    data = np.array(list_points)
    try:
        # Fourier-Transformation
        fourier_transform = np.fft.fft(data)
        #frequencies = np.fft.fftfreq(len(fourier_transform), t[1]-t[0])
        frequencies = np.fft.fftfreq(len(fourier_transform), 1)
        # Finden der dominierenden Frequenz (Peak)
        peaks, _ = find_peaks(np.abs(fourier_transform))
        dominante_frequenz_index = peaks[np.argmax(np.abs(fourier_transform[peaks]))]
        dominante_frequenz = frequencies[dominante_frequenz_index]
    except:
        return 0, 0
    
    periodendauer = 1 / dominante_frequenz  
    ziegel = len(list_points) / (periodendauer**2)**0.5
    
    if False:
        # Plot der Fourier-Transformierten
        plt.plot(frequencies, np.abs(fourier_transform))
        plt.plot(frequencies[dominante_frequenz_index], np.abs(fourier_transform[dominante_frequenz_index]), 'rx')
        plt.title('Amplituden Spektrum ' + "_" + str(ziegel))
        plt.xlabel('Frequenz')
        plt.ylabel('Amplitude')
        plt.show()
        # Plot des Arrays
        plt.plot(data)
        plt.title('Histogramschwingung ' + "_" + str(ziegel))
        plt.xlabel('Zeit')
        plt.ylabel('Amplitude')
        plt.show()
        print(f"Die dominante Frequenz (Grundschwingung) beträgt {dominante_frequenz} Hz.")
        print(f"Es wurden {ziegel} Ziegel mittels Fourier bestimmt")
    return periodendauer, ziegel

def increase_contrast(image, alpha, beta):
    """Increase the contrast of the input image.

    Parameters:
    - image: Input image
    - alpha: Contrast control (1.0-3.0, 1.0 means no change)
    - beta: Brightness control (0-100, 0 means no change)

    Returns:
    - Contrast-enhanced image
    """
    new_image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)
    return new_image

def calculate_list(list_1, list_2, operand):
    """This function calculates two lists
    Parameters:
        - list 1
        - list 2
        - operand as string
        
    Returns:
        - result list
    """
    result_list = []
    for a, b in zip(list_1, list_2):
        try:
            if operand == "+":
                result_list.append(a + b)
            if operand == "-":
                result_list.append(a - b)
            if operand == "/":
                result_list.append(a / b)
            if operand == "*":
                result_list.append(a * b)
        except:
            result_list.append("f")
    return result_list

def extract_vertical_lines(img):
    kernel = np.ones((7, 3), np.uint8)
    vertical_lines = cv2.erode(img, kernel, iterations=1)
    return vertical_lines

def find_non_type_in_list(original_list):
    """ This list finds non_type elements in list and replaces it with 0
    Parameters:
        - original_list
    
    Return:
        - modified list
    """
    modified_list = [0 if not isinstance(item, (int, float)) else item for item in original_list]
    return modified_list

def get_histogram(image, cut_factor):
    """This function calculates the average row shades of an image
    Parameters:
        - image
        - cut factor: cuts rows to begin of list and at the end
    Return:
        - average row shades
    """
    # Get the dimensions of the image
    height, width = image.shape
    
    # Initialize an array to store the average row colors
    average_row_shades = []
    
    # Iterate through each row
    for z in range(height):
        row = image[z, :]
        average_shade = np.mean(row)
        average_row_shades.append((average_shade))
    if cut_factor > 0:
        average_row_shades = average_row_shades[cut_factor:]
        average_row_shades = average_row_shades[:len(average_row_shades) - cut_factor]

    return average_row_shades
	
def sharpen_image(image_path):
    # Bild einlesen
    img = cv2.imread(image_path)

    # Schärfen Kernel erstellen
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])

    # Bild schärfen
    neues_bild = cv2.filter2D(img, -1, kernel)

    return neues_bild

def find_slices_from_roof(verzeichnis_pfad, suchtext):
    """ This function finds the slices of the roofnumber
    Parameter:
        - verzeichnis_pfad 
        - suchtext example(1_22_01_69 suchtext = "1_")
    Return:
        - list with slices
    """
    # Suche nach Dateien im Verzeichnis
    dateien = os.listdir(verzeichnis_pfad)
    
    # Filtere Dateien, die mit "1_" beginnen
    passende_dateien = [datei for datei in dateien if datei.startswith(suchtext)]
    return passende_dateien

def split_image_into_strips(image_slice, num_strips):
    #image_slice = np.rot90(image_slice)
    height, width = image_slice.shape[:2]
    strip_height = height // num_strips

    strips = []
    for i in range(num_strips):
        start = i * strip_height
        end = start + strip_height
        strip = image_slice[start:end, :]
        strips.append(strip)

        # Speichern des Streifens mit Namenskonvention
        #slice_number = str(i + 1).zfill(2)
        #output_path = '/content/drive/MyDrive/Masterprojekt/Datensatz/slices_train_y_axis/' + f'{roof_number}_{num_strips}_{slice_number}_{max_x}.png'
        #cv2.imwrite(output_path, strip)
    return strips

def save_slices(slices):
    i = 0
    for element in slices:
        output_path = r"C:\Users\ibit\OneDrive\Desktop\temp\slices\\" + str(i) + ".png"
        strip = element
        for j in range(3):
            strip = np.rot90(strip) #Bild um 90° gedreht
        cv2.imwrite(output_path, strip)
        i += 1
        
    
def finde_haeufigster_wert(liste):
    # Verwende Counter, um die Häufigkeit der Elemente zu zählen
    zaehler = Counter(liste)

    # Finde die maximale Häufigkeit
    max_haeufigkeit = max(zaehler.values())

    # Finde alle Werte mit der maximalen Häufigkeit
    haeufigste_werte = [wert for wert, haeufigkeit in zaehler.items() if haeufigkeit == max_haeufigkeit]
    return haeufigste_werte

def loss_prediction_x(df):
    df = df.loc[df["Dachnummer"] > 139]
    df = df.loc[df["Abweichung X"] > -3 and df["Abweichung X"] < 3]
    df = df.loc[df["Abweichung Y"] > -3 and df["Abweichung Y"] < 3]
    for col in df.columns:
        df = df[~df[col].apply(is_letter)] #Zeilen, die Buchstaben enthalten löschen
    return

def is_letter(value):
    return isinstance(value, str) and value.isalpha()

def pad_dict_list(dict_list, padel):
    lmax = 0
    for lname in dict_list.keys():
        lmax = max(lmax, len(dict_list[lname]))
    for lname in dict_list.keys():
        ll = len(dict_list[lname])
        if  ll < lmax:
            dict_list[lname] += [padel] * (lmax - ll)
    return dict_list

def sharpen_image_x(image, sharpen_factor):

    # Laplace-Operator für die Kantenextraktion
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # Formel zur Anpassung der Schärfe
    adjusted = image - sharpen_factor * laplacian

    # Skalierung der Werte auf den Bereich von 0 bis 255
    adjusted = np.clip(adjusted, 0, 255)
    image_x = np.uint8(adjusted)
    cv2.imshow('Sharpened', image_x)
    return image_x

def sharpen_image_y(image, sharpen_factor):

    # Laplace-Operator für die Kantenextraktion
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # Formel zur Anpassung der Schärfe
    adjusted = image - sharpen_factor * laplacian

    # Skalierung der Werte auf den Bereich von 0 bis 255
    adjusted = np.clip(adjusted, 0, 255)
    image_y = np.uint8(adjusted)
    cv2.imshow('Sharpened', image_y)
    return image_y

def extrahiere_x_aus_dateiname(dateiname):
    # Verwende regulären Ausdruck, um die Zahl zwischen "gewarptes_bild" und ".png" zu extrahieren
    match = re.search(r'gewarptes_bild(\d+)\.png', dateiname)
    
    if match:
        # Gibt den gefundenen Wert von x zurück
        return match.group(1)
    else:
        # Gibt None zurück, wenn keine Übereinstimmung gefunden wurde
        return None

#######################################################################################################################################################
def get_dachziegel_x_y(image):
    #Faktoren ermittelt aus Sensitivitätsanalyse für Bilder Dimension 700x300 
    slice_factor = 4
    cut_factor_vertical = 0
    cut_factor_horizontally = 0
    fail = False
    
    max_ziegel_dach_x = 10
    max_ziegel_dach_y = 10
    average_row_shades_x = []
    average_row_shades_y = []
    
    image_y = image
    image_x = image_y

    # Convert the image to grayscale
    image_y = cv2.cvtColor(image_y, cv2.COLOR_BGR2GRAY)
    image_x = cv2.cvtColor(image_x, cv2.COLOR_BGR2GRAY)     

    average_row_shades_y = get_histogram(image_y, cut_factor_vertical)
    periodendauer_y, ziegel_y = fourier_from_list(average_row_shades_y)
    ziegel_y = round(ziegel_y+1)
    if ziegel_y < max_ziegel_dach_y:
        print("keine Voraussage moeglich")
        fail = True
    splits = split_image_into_strips(image_x, int(round(ziegel_y, 0)) * slice_factor)
    slices_ziegel_x = []
    for i in range(0, len(splits)):
        splits[i] = np.rot90(splits[i]) #Bild um 90° gedreht
        average_row_shades_x = get_histogram(splits[i], cut_factor_horizontally)
        periodendauer_x, ziegel_x = fourier_from_list(average_row_shades_x)
        slices_ziegel_x.append(round(ziegel_x, 2))
    if len(finde_haeufigster_wert(slices_ziegel_x)) == 1 and finde_haeufigster_wert(slices_ziegel_x)[0] > max_ziegel_dach_x: #mehrere Werte gleich oft ermittelt               
        ziegel_x = round(finde_haeufigster_wert(slices_ziegel_x)[0]+1)
    else:
        print("keine Voraussage moeglich")
        fail = True
    if fail:
        ziegel_x = 0
        ziegel_y = 0
    return ziegel_x, ziegel_y
def open_cv_window():
    global cv_image, current_filename

    # Bildpfad auswählen
    bildpfad = filedialog.askopenfilename()
    if not bildpfad:
        return

    # Extrahiere den ursprünglichen Dateinamen aus dem ausgewählten Bildpfad
    current_filename = bildpfad.split('/')[-1].split('.')[0]

    # Bild mit OpenCV laden
    cv_image = cv2.imread(bildpfad)

    # Bildmitte berechnen
    h, w, _ = cv_image.shape
    mitte_x, mitte_y = w // 2, h // 2

    # Punkte um die Bildmitte zentrieren
    verschiebung_x = 100  # Verschiebung vom Zentrum für den Startpunkt
    verschiebung_y = 100
    punkte = np.array([(mitte_x-verschiebung_x, mitte_y-verschiebung_y), 
                       (mitte_x+verschiebung_x, mitte_y-verschiebung_y), 
                       (mitte_x+verschiebung_x, mitte_y+verschiebung_y), 
                       (mitte_x-verschiebung_x, mitte_y+verschiebung_y)], dtype=np.int32)

    moving_point = None

    def draw_points_and_lines(image, points):
        # Zeichne Linien zwischen den Punkten
        cv2.polylines(image, [points], isClosed=True, color=(0,255,0), thickness=2)
        # Zeichne die Punkte
        for point in points:
            cv2.circle(image, tuple(point), 5, (0, 0, 255), -1)

    def mausklick(event, x, y, flags, param):
        global moving_point
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, point in enumerate(punkte):
                if abs(x - point[0]) < 10 and abs(y - point[1]) < 10:
                    moving_point = i

        elif event == cv2.EVENT_MOUSEMOVE and moving_point is not None:
            punkte[moving_point] = [x, y]

        elif event == cv2.EVENT_LBUTTONUP:
            moving_point = None

    cv2.namedWindow("Roof area selection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Roof area selection", 800, 600)  # Setzt die Fenstergröße auf 800x600
    cv2.setMouseCallback("Roof area selection", mausklick)

    while True:
        img_copy = cv_image.copy()
        draw_points_and_lines(img_copy, punkte)
        cv2.imshow("Roof area selection", img_copy)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter-Taste
            break

    cv2.destroyAllWindows()
    process_image(punkte) 



def process_image(punkte):
    global photo, canvas, gewarptes_bild, fourier_button, cv_image

    # Extrahiere x- und y-Werte separat und berechne Pixelabmessungen
    x_values = [point[0] for point in punkte]
    y_values = [point[1] for point in punkte]
    min_x, max_x, min_y, max_y = min(x_values), max(x_values), min(y_values), max(y_values)
    pixelbreite_b, pixelhoehe_b = max_x - min_x, max_y - min_y

    # Erstelle eine leere Maske und zeichne das Viereck
    mask = np.zeros(cv_image.shape[:2], dtype=np.uint8)
    punkte_arr = np.array(punkte, dtype=np.int32)
    cv2.fillPoly(mask, [punkte_arr], 255)

    # Schneide den ausgewählten Bereich aus und mache ihn transparent
    ausgeschnittener_bereich = cv2.bitwise_and(cv_image, cv_image, mask=mask)
    ausgeschnittener_bereich_rgba = cv2.cvtColor(ausgeschnittener_bereich, cv2.COLOR_BGR2BGRA)
    ausgeschnittener_bereich_rgba[np.where((ausgeschnittener_bereich == [0, 0, 0]).all(axis=2))] = [0, 0, 0, 0]

    # Warping des ausgeschnittenen Bereichs
    zielkoordinaten = np.array([[0, 0], [pixelbreite_b, 0], [pixelbreite_b, pixelhoehe_b], [0, pixelhoehe_b]], dtype=np.float32)
    transformationsmatrix = cv2.getPerspectiveTransform(punkte_arr.astype(np.float32), zielkoordinaten)
    gewarptes_bild = cv2.warpPerspective(ausgeschnittener_bereich_rgba, transformationsmatrix, (pixelbreite_b, pixelhoehe_b))

    # Skalierung des Bildes, um es vollständig im Canvas anzuzeigen
    canvas_width, canvas_height = canvas.winfo_width(), canvas.winfo_height()
    scale_width = canvas_width / gewarptes_bild.shape[1]
    scale_height = canvas_height / gewarptes_bild.shape[0]
    scale_factor = min(scale_width, scale_height)

    # Anpassen der Größe des gewarpten Bildes
    new_width = int(gewarptes_bild.shape[1] * scale_factor)
    new_height = int(gewarptes_bild.shape[0] * scale_factor)
    gewarptes_bild_resized = cv2.resize(gewarptes_bild, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Anzeigen des bearbeiteten und skalierten Bildes im Tkinter-Fenster
    pil_image = Image.fromarray(cv2.cvtColor(gewarptes_bild_resized, cv2.COLOR_BGRA2RGBA))
    photo = ImageTk.PhotoImage(image=pil_image)
    canvas.create_image((canvas_width - new_width) // 2, (canvas_height - new_height) // 2, image=photo, anchor=tk.NW)

    fourier_button.config(state=tk.NORMAL)
    open_cv_button.pack_forget()



def display_fourier_results():
    # Ziegel zählen
    ziegel_x, ziegel_y = get_dachziegel_x_y(gewarptes_bild)
    results_label.config(text=f"Number of tiles in X-direction: {ziegel_x}\nNumber of tiles in Y-direction: {ziegel_y}")

def reset_app():
    global cv_image, photo, canvas

    # Reset der globalen Variablen
    cv_image = None
    photo = None

    # Canvas und Labels zurücksetzen
    canvas.delete("all")
    results_label.config(text="")

    # Buttons zurücksetzen
    fourier_button.config(state=tk.DISABLED)

def show_info_window():
    # Neues Fenster für die Info erstellen
    info_window = Toplevel(main_window)
    info_window.title("Info")

    # Bild laden und anzeigen
    image = Image.open("Beispiel.jpg")  # Pfad zu Ihrem Bild
    photo = ImageTk.PhotoImage(image)
    image_label = tk.Label(info_window, image=photo)
    image_label.image = photo  # Referenz halten, um GC zu vermeiden
    image_label.pack()

    # Text anzeigen
    info_text = "Für das richtige Erfassen der Dachfläche ist es nötig, dass die Mitte der äußersten Eckziegel erfasst wird. Das Beispielbild zeigt wie die Punkte auf dem Dach gesetzt werden sollen. "
    text_label = tk.Label(info_window, text=info_text, wraplength=400)
    text_label.pack()

main_window = tk.Tk()
main_window.title("Rooftile Counting Tool")
main_window.geometry("1000x550")

# Ein Frame für die Inhalte des Hauptfensters erstellen
content_frame = tk.Frame(main_window)
content_frame.pack(expand=True, fill=tk.BOTH)  # Das Frame füllt das gesamte Hauptfenster und passt sich der Größe an

# Canvas für das bearbeitete Bild, in grid positioniert
canvas = tk.Canvas(content_frame, bg='grey')  # Hintergrundfarbe nur zur Veranschaulichung
canvas.grid(row=0, column=0, columnspan=3, sticky="nsew")

# Konfiguration der Zeilen und Spalten im content_frame
content_frame.rowconfigure(0, weight=1)  # Erlaubt dem Canvas, sich in der Vertikalen zu dehnen
for i in range(3):
    content_frame.columnconfigure(i, weight=1)  # Erlaubt den Spalten, sich gleichmäßig zu dehnen

# Buttons und Label platzieren
fourier_button = tk.Button(content_frame, text="Count rooftiles", command=display_fourier_results, state=tk.DISABLED)
fourier_button.grid(row=1, column=0, sticky="ew",padx= 5)

results_label = tk.Label(content_frame, text="")
results_label.grid(row=2, column=1, sticky="ew")

open_cv_button = tk.Button(content_frame, text="Upload image and select roof area", command=open_cv_window)
open_cv_button.grid(row=1, column=2, sticky="ew" ,padx= 5)

# Info- und Reset-Buttons in separaten Frames
info_frame = tk.Frame(content_frame)
info_frame.grid(row=2, column=2, sticky="ne", padx=20, pady=20)

reset_frame = tk.Frame(content_frame)
reset_frame.grid(row=2, column=0, sticky="nw", padx=20, pady=20)

info_button = tk.Button(info_frame, text="Info", command=show_info_window)
info_button.pack()

reset_button = tk.Button(reset_frame, text="Reset", command=reset_app)
reset_button.pack()

# Stellt sicher, dass das content_frame sich mit dem Fenster mitdehnt
main_window.rowconfigure(0, weight=1)
main_window.columnconfigure(0, weight=1)

main_window.mainloop()