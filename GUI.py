import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import csv,sys
import threading
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

import seaborn as sns
import random
from numba import jit
import os
import time
import queue
directory = "./images/"
if not os.path.exists(directory):
    os.makedirs(directory)
def genname(parameters):
    name=''
    for i in range(len(parameters)-1):
        p=parameters[i]
        if p=='Centeral\n Cell':
            name+='0'
        elif p=='Middle\n Third':
            name+='1'
        elif p=='Entire\n Chain':
            name+='2'
        elif p==True:
            name+='1'
        elif p== False:
            name+='0'
        elif float(p)%1==0:
            p=int(p)
            if p<0:
                name+=str(777)
            name+=str(abs(p))
        else:
            name+=str(abs(int(p)//1))
            name+=str(888)
            name+=str(abs(float(p)%1))[2:]
    return base10_to_base64(int(name))


def base10_to_base64(num):
    allowed_chars = [chr(i) for i in range(33, 256) if (chr(i) not in '<>:\"/\\|?*' and chr(i)!=' ')]
    chars=''.join(allowed_chars)
    base64_chars =chars
    base64_str = ""
    while num > 0:
        remainder = num % 214
        base64_str = base64_chars[remainder] + base64_str
        num = num // 214
    print(base64_str)
    return "_"+base64_str
def Generate_Plot(parameters, para2):
    """Generates a plot based on input parameters and additional information."""
    
    # Printing additional parameter information
    print(para2)
    
    # Reformat para2 dictionaries, removing newline characters in string values
    tuple_list = [(k, v.replace("\n", "") if isinstance(v, str) else v) for dictionary in para2 for k, v in dictionary.items()]
    
    # Extracting parameters
    qtype, ptype, mappings, number_of_maps, coupling_range, k, g, den, Q0, P0, delta, samples, show_values, _, plotname = parameters
    mappings = int(mappings)
    number_of_maps = int(number_of_maps)
    coupling_range = int(coupling_range)
    delta = 10 ** int(delta)
    samples = int(samples)
    k = float(k)
    g = float(g)
    den = float(den)
    Q0 = float(Q0)
    P0 = float(P0)
    
    directory = ""  # Define directory if needed
    plot_file_path = directory + plotname + ".png"
    
    # If the plot already exists, no need to regenerate it
    if any(file == plotname + ".png" for file in os.listdir(directory)):
        print("Image Already Exists: Skipping Generation")
        return plot_file_path
    
    # Constructing the matrix (M)
    print("Generating Matrix (M): 0.00%", end='\r')
    n = number_of_maps
    K = k * np.eye(n)
    G = g * np.eye(min(coupling_range, n))
    C = K + np.sum([G[l] * (np.eye(n, k=-l-1) + np.eye(n, k=l+1)) for l in range(min(coupling_range, (n - 1) // 2))], axis=0)
    
    M = np.block([[np.eye(n), C], [C, C @ C + np.eye(n)]])
    print("Generating Matrix (M): 100.00%")
    
    # Applying mappings to the matrix and generating the plot data
    plot_data = apply_mappings_and_generate_data(M, Q0, delta, samples, mappings, qtype, ptype, n, den)
    
    # Constructing the title string for the plot
    title_string = construct_title_string(tuple_list)
    
    return title_string, plot_data, show_values, True, n, plot_file_path


def apply_mappings_and_generate_data(M, Q0, delta, samples, mappings, qtype, ptype, n, den):
    """Applies mappings to the matrix M and generates the plot data."""
    
    Qsum = np.zeros((mappings, n))
    for it in range(samples):
        q0 = initialize_q0(Q0, delta, den, n, qtype)
        p0 = initialize_p0(P0, delta, den, n, ptype)
        x = np.concatenate([q0, p0])
        for j in range(mappings):
            if (it*mappings+j)%int(mappings*samples/100+1)==0:
                print("Sampling & Mapping: "+str(round(100*(it*mappings+j)/(mappings*samples),2))+"%",end='\r')
            Qsum[j] += x[:n]
            x = np.mod(M @ x, den)
    print("Sampling & Mapping: 100.00%")
    
    return np.vstack([np.zeros((1, mappings)), Qsum.T / (samples * den)])


def initialize_q0(Q0, delta, den, n, qtype):
    """Initializes the q0 array based on the type specified."""
    
    q0 = np.zeros(n)
    if qtype == 'Centeral\n Cell':
        q0[(n-1)//2] = (Q0 + delta*den*(random.random()-0.5))%den
    elif qtype == 'Middle\n Third':
        q0[(n-1)//3+1:2*(n-1)//3+1] = (Q0 + delta*den*(random.random()-0.5))%den
    elif qtype == 'Entire\n Chain':
        q0 = np.ones(n)*(Q0 + delta*den*(random.random()-0.5))%den
    return q0


def initialize_p0(P0, delta, den, n, ptype):
    """Initializes the p0 array based on the type specified."""
    
    p0 = np.zeros(n)
    if ptype == 'Centeral\n Cell':
        p0[(n-1)//2] = (P0 + delta*den*(random.random()-0.5))%den
    elif ptype == 'Middle\n Third':
        p0[(n-1)//3+1:2*(n-1)//3+1] = (P0 + delta*den*(random.random()-0.5))%den
    elif ptype == 'Entire\n Chain':
        p0 = np.ones(n)*(P0 + delta*den*(random.random()-0.5))%den
    return p0


def construct_title_string(tuple_list):
    """Constructs the title string for the plot based on input parameters."""
    
    values_box = "|"
    for i in range(1, len(tuple_list)):
        value_add = "    " + tuple_list[i][0] + ": " + str(tuple_list[i][1]) + "   |"
        if len(values_box.split('\n')[-1]) + len(value_add) > 50:
            values_box += '\n|'
        values_box += value_add
    return values_box

    #return()




class StdoutRedirect:
    def __init__(self, text_widget):
        self.text_space = text_widget
        self.last_was_cr = False
        
    def write(self, string):
        if self.last_was_cr:
            # delete last line
            self.text_space.delete("end-2l", "end-1c")
            
        if string.endswith('\flush'):
            self.text_space.delete("1.0", tk.END)
            string = string.rstrip('\flush')
            
        # if this message ends with a carriage return, don't add a newline
        self.last_was_cr = string.endswith("\r")
        if self.last_was_cr:
            string = string.rstrip("\r")

        self.text_space.insert(tk.END, string + ("" if not self.last_was_cr else "\n"))
        self.text_space.see(tk.END)

    def flush(self):
        pass


types=[0,1,1,2,2,2,2,2,2,2,2,2,2,0,3]

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.data_queue = queue.Queue()
        self.master.config(height=1000)
        self.grid(sticky='news')
        self.plot_lock = threading.Lock()
        # Default values for each column
        self.default_values = [True, '','', 10,29, 1,1, 1,10, 1,0, -999,1, True,'plot' ]
        self.create_widgets()
        # Define scrollbars for the Figure Frame

        sys.stdout = StdoutRedirect(self.console_output)
    
    
    ######################
    # Clear Table Section
    ######################
    def clear_table(self):
        # Destroy each widget in each row
        for row in self.rows:
            for entry in row:
                entry.destroy()

        # Clear the rows list
        self.rows.clear()

    ##############################
    # Export to CSV File Section
    ##############################
    def export_to_csv(self):
        filename = self.csv_filename_entry.get()
        if not filename:
            print("Please enter a filename.")
            return

        data = []
        for row in self.rows:
            row_data = {}
            for i, entry in enumerate(row):
                if types[i]==0:
                    row_data[self.headers[i]] = entry.var.get()
                if types[i]==1:
                    row_data[self.headers[i]] = entry.var.get()
                if types[i]==2:
                    row_data[self.headers[i]] = entry.get()
                if types[i]==3:
                    row_data[self.headers[i]] = entry.get()
            data.append(row_data)

        try:
            with open(filename+".csv", 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.headers)
                writer.writeheader()
                for row_data in data:
                    writer.writerow(row_data)
            print("Export Successful")
        except Exception as e:
            print("Could not write to file:", e)

    ##############################
    # Import from CSV File Section
    ##############################
    def import_from_csv(self):
        filename = self.csv_filename_entry.get()
        if not filename:
            print("Please enter a filename.")
            return

        self.clear_table()

        try:
            with open(filename+".csv", 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    self.add_row()
                    row_entries = self.rows[-1]
                    for i, header in enumerate(self.headers):
                        if types[i]==0:
                            check_var = row_entries[i].var
                            check_var.set(row[header] == 'True')
                        if types[i]==1:
                            option_var = row_entries[i].var
                            option_var.set(row[header])
                        if types[i]==2:
                            row_entries[i].delete(0, tk.END)
                            row_entries[i].insert(0, row[header])
                        if types[i]==3:
                            row_entries[i].delete(0, tk.END)
                            row_entries[i].insert(0, row[header])   
        except Exception as e:
            print("Could not read from file:", e)

    ######################
    # Create Widgets Section
    ######################
    def create_widgets(self):
        # Headers for each column
        self.headers = ['Active','Q_0 Pertubation Type','P_0 Pertubation Type','Mappings (m)', 
                        'Number of Maps (n)', 'Coupling Range (l)', 'K-Fib (K)', 'Coupling Strength (G)', 
                        'Denominator', 'Q0', 'P0', 'Delta exponent', '# Samples', 'Show values on Graph', 'Plot Name']

        # Button Frame: Contains all buttons like Add Column, Run All, Export to CSV etc.
        
        # Table Frame: Contains a table of data entries.
        self.table_frame = tk.Frame(self)
        self.table_frame.grid(row=0, column=0, sticky='news')
        self.button_frame = tk.Frame(self.table_frame)
        #self.button_frame.grid(row=1, column=0, columnspan=2, sticky='news')  # spanned the button frame to occupy both columns
        self.button_frame.grid(row=2, column=0,sticky='news')
        # Figure Frame: This will contain the plot figure
        self.fig_frame = tk.Frame(self)
        self.fig_frame.grid(row=0, column=1, sticky='news')

        # configure the grid to distribute extra space equally
        #self.grid_columnconfigure(0, weight=1)  
        #self.grid_columnconfigure(1, weight=1)
        #self.grid_rowconfigure(0, weight=1)
        # give the button frame less weight since it should not expand as much as the others
        #self.grid_rowconfigure(1, weight=0) 

        # Canvas for the Table Frame
        self.table_canvas = tk.Canvas(self.table_frame)
        self.table_scrollbar_y = tk.Scrollbar(self.table_frame, orient='vertical', command=self.table_canvas.yview)
        self.table_scrollbar_x = tk.Scrollbar(self.table_frame, orient='horizontal', command=self.table_canvas.xview)

        # Scrollable Frame for the Table Frame: To enable scrolling in case the table gets too large.
        self.scrollable_table_frame = ttk.Frame(self.table_canvas)
        self.scrollable_table_frame.bind('<Configure>', lambda e: self.table_canvas.configure(scrollregion=self.table_canvas.bbox('all')))
        self.table_canvas.create_window((0, 0), window=self.scrollable_table_frame, anchor='nw')
        self.table_canvas.configure(yscrollcommand=self.table_scrollbar_y.set, xscrollcommand=self.table_scrollbar_x.set,height=620)
        self.table_canvas.grid(row=0, column=0, sticky='news')
        self.table_scrollbar_y.grid(row=0, column=1, sticky='ns')
        self.table_scrollbar_x.grid(row=1, column=0, sticky='ew')

        # Canvas for the Figure Frame
        self.fig_canvas = tk.Canvas(self.fig_frame)
        self.fig_scrollbar_y = tk.Scrollbar(self.fig_frame, orient='vertical', command=self.fig_canvas.yview)
        self.fig_scrollbar_x = tk.Scrollbar(self.fig_frame, orient='horizontal', command=self.fig_canvas.xview)

        # Scrollable Frame for the Figure Frame: To enable scrolling in case the table gets too large.
        self.scrollable_fig_frame = ttk.Frame(self.fig_canvas)
        self.scrollable_fig_frame.bind('<Configure>', lambda e: self.fig_canvas.configure(scrollregion=self.fig_canvas.bbox('all')))
        self.fig_canvas.create_window((0, 0), window=self.scrollable_fig_frame, anchor='nw')
        self.fig_canvas.configure(yscrollcommand=self.fig_scrollbar_y.set, xscrollcommand=self.fig_scrollbar_x.set,height=750,width=600)
        self.fig_canvas.grid(row=0, column=0, sticky='news')
        self.fig_scrollbar_y.grid(row=0, column=1, sticky='ns')
        self.fig_scrollbar_x.grid(row=1, column=0, sticky='ew')

        self.button_and_image_frame = tk.Frame(self.fig_frame)
        self.button_and_image_frame.grid(row=0, column=0, sticky='sw')

        self.image_frame = tk.Frame(self.button_and_image_frame)
        self.image_frame.grid(row=1, column=0)

        self.zoom_button_frame = tk.Frame(self.button_and_image_frame)
        self.zoom_button_frame.grid(row=0, column=0)

        self.zoom_in_button = tk.Button(self.zoom_button_frame, text='Zoom In', command=self.zoom_in)
        self.zoom_in_button.grid(row=0, column=0)

        self.zoom_out_button = tk.Button(self.zoom_button_frame, text='Zoom Out', command=self.zoom_out)
        self.zoom_out_button.grid(row=0, column=1)

        
        # Headers
        for i, header in enumerate(self.headers):
            tk.Label(self.scrollable_table_frame, text=header).grid(row=i, column=0)
        self.run_all_button = tk.Button(self.scrollable_table_frame, text="Run All", command=self.run_all).grid(row=15, column=0)
        tk.Label(self.scrollable_table_frame, text='').grid(row=17, column=0)
        tk.Label(self.scrollable_table_frame, text='cvs file name for import/export').grid(row=18, column=1, columnspan=2)
        # CSV Filename Entry Field
        self.csv_filename_entry = tk.Entry(self.scrollable_table_frame, width=15)
        self.csv_filename_entry.grid(row=19, column=1, sticky='news', columnspan=2)


        self.export_button = tk.Button(self.scrollable_table_frame, text="Export .csv", command=self.export_to_csv).grid(row=20, column=1, columnspan=1)
        self.import_button = tk.Button(self.scrollable_table_frame, text="Import .csv", command=self.import_from_csv).grid(row=20, column=2,sticky='w', columnspan=1)
        self.add_row_button = tk.Button(self.scrollable_table_frame, text="Add Column", command=self.add_row).grid(row=19, column=0)
        self.clear_table_button = tk.Button(self.scrollable_table_frame, text="Clear Table", command=self.clear_table).grid(row=20, column=0)

        self.console_output = tk.Text(self.button_frame, wrap='word',height=5,width=10)
        self.console_output.grid(row=1,sticky='n')
        self.console_output.pack(fill='both', expand=True)

        # Rows of data entry fields. Initially empty.
        self.rows = []
        self.add_row()
        
    ######################
    # Add Row Section
    ######################
    def add_row(self):
        # Add a new row
        row = []
        for i in range(len(self.headers)):
            if types[i] == 0:  # Checkbutton
                check_var = tk.BooleanVar()
                entry = tk.Checkbutton(self.scrollable_table_frame, variable=check_var)
                entry.toggle()
                entry.grid(row=i, column=len(self.rows) + 1)
                entry.var = check_var  # Store the BooleanVar object
            elif types[i] == 1:  # OptionMenu
                option_var = tk.StringVar(value="Centeral\n Cell")
                options = ["Centeral\n Cell", "Middle\n Third","Entire\n Chain"]
                entry = tk.OptionMenu(self.scrollable_table_frame, option_var, *options)
                entry.grid(row=i, column=len(self.rows) + 1)
                entry.var = option_var  # Store the StringVar object
            elif types[i]==2:  # Entry
                entry = tk.Entry(self.scrollable_table_frame,width=12)
                entry.insert(0, self.default_values[i])
                entry.grid(row=i, column=len(self.rows) + 1)
            elif types[i]==3:
                entry = tk.Entry(self.scrollable_table_frame,width=12)
                entry.insert(0, self.default_values[i])
                entry.grid(row=i, column=len(self.rows) + 1)
            row.append(entry)
        run_button = tk.Button(self.scrollable_table_frame)
        run_button["text"] = "Run"
        run_button["command"] = lambda: self.run_script(row)
        run_button.grid(row=len(self.headers), column=len(self.rows) + 1)
        self.rows.append(row)
    def generate_plot(self, data):
        # Unpack the data

        try:
            (values_box, plot1, show_values, show_cbar, n, filepath) = data
        except:
            return data
        plt.close()

        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [5, 1]}, figsize=(8, 10))

        sns.heatmap(plot1,annot=show_values,cmap="turbo", linewidths=0, ax=ax1, cbar=show_cbar,vmin=0, vmax=1,cbar_kws={'label': '(Average) position value q within cell'})

        ax1.set_xlabel("$m^{th}$ mapping",size=16)
        ax1.set_ylabel("$i^{th}$ cell in the chain",size=16)
        ax1.set_ylim(1,n+1)
        ax2.text(0.5, 0.5, values_box, ha='center', va='center', wrap=True,size=11)
        ax2.axis('off')
        plt.tight_layout()

        fig.savefig(filepath,dpi=256)
        return filepath

    def display_plot(self, fig):
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        img = Image.open(fig)
        self.original_image = img
        self.zoom_level = 1
        self.master.after(1, self.update_image)
        print("Done")
        return
    def check_queue(self):
        try:
            plot_data = self.data_queue.get_nowait()  # Non-blocking get
            if plot_data is not None:
                self.display_plot(self.generate_plot(plot_data))
            self.master.after(100, self.check_queue)  # Check again after 100 ms
        except queue.Empty:
            # If the queue was empty, check again after 100 ms
            self.master.after(100, self.check_queue)
    def run_script(self, row):
        data = []
        print(end='\flush')

        row_data1 = {}
        for i, entry in enumerate(row):
            if types[i]==0:
                row_data1[self.headers[i]] = entry.var.get()
            if types[i]==1:
                row_data1[self.headers[i]] = entry.var.get()
            if types[i]==2:
                row_data1[self.headers[i]] = entry.get()
            if types[i]==3:
                row_data1[self.headers[i]] = entry.get()
        data.append(row_data1)

        row_data = []
        for i, entry in enumerate(row):
            if types[i]==0:
                row_data.append(entry.var.get())
                if i==0 and row_data[i]==False:
                    print("Not Active")
                    return
                            
            if types[i]==1:
                row_data.append(entry.var.get())
            if types[i]==2:
                row_data.append(float(entry.get()))
            if types[i]==3:
                row_data.append(entry.get())
        
        thread = threading.Thread(target=self.generate_plot_thread, args=(row_data,data))
        thread.start()
        # Put None as a signal for end of processing
        self.data_queue.put(None)  # Signal for end of processing
        self.check_queue()  # Start polling the queue
    def generate_plot_thread(self, row_data, data):
        with self.plot_lock:
            plot_data = Generate_Plot(row_data, data)
            self.data_queue.put(plot_data)

    def update_image(self):
        if not self.fig_canvas.winfo_exists():
            return

        img = self.original_image.resize((int(600 * self.zoom_level), int(750 * self.zoom_level)))
        photo = ImageTk.PhotoImage(img)

        self.fig_canvas.delete('all')
        self.fig_canvas.config(scrollregion=(0, 0, img.width, img.height))
        self.fig_canvas.create_image(0, 0, image=photo, anchor='nw') 

        self.fig_canvas.image = photo

    def zoom_in(self):
        self.zoom_level *= 1.2
        self.update_image()

    def zoom_out(self):
        self.zoom_level /= 1.2
        self.update_image()

    def run_all(self):
        for row in self.rows:
            self.run_script(row)
        # Signal that there's no more data
        

        
            

######################
# Main Section
######################
import ctypes
 
ctypes.windll.shcore.SetProcessDpiAwareness(1)
root = tk.Tk()
root.geometry("1200x800")
root.title("1D Chain Propegation")
app = Application(master=root)
app.mainloop()
##if i==0:
##    var= tk.BooleanVar()
##    entry=tk.Checkbutton(self)
##else:
##    entry = tk.Entry(self)
##entry.grid(row=len(self.rows)+1, column=i)
##row.append(entry)
#self.headers = ['Active','Mappings', 'Number of Maps', 'Coupling Range', 'K-Fib', 'Coupling Strength', 'Den', 'Q0', 'P0', '$\Delta$', 'samples', 'show values', 'plot name']



