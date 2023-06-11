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
            #print(str(abs(float(p)%1)))
            if str(abs(float(p)%1))[1]=='e':
                name+=str(abs(float(p)%1))[3:]
            else:
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

def generate_matrix_C_2D(n, k, g,coupling_range):
    K = k * np.eye(n**2)
    G = g * np.eye(n**2)
    P = np.roll(np.eye(n), 1, axis=1)
    C=K
    for l in range(1,(n + 1) // 2):
        if l>coupling_range:
            break
        P = np.roll(np.eye(n), l, axis=1)
        P_kron_I = np.kron(P, np.eye(n))
        I_kron_P = np.kron(np.eye(n), P)
        C += (P_kron_I + I_kron_P) @ G + G @ (P_kron_I.T + I_kron_P.T)
    return C

@jit(nopython=True)
def generate_initial_condition(n, Q0, P0, delta):
    q = np.zeros((n, n))
    p = np.zeros((n, n))
    q[n//2, n//2] = Q0 + (0.5 - np.random.random()) * delta
    p[n//2, n//2] = P0
    x = np.concatenate((q.flatten(), p.flatten()))
    return x

@jit(nopython=True)
def iterate(M, x,den, num_iterations, n):
    q_avg = np.zeros((n, n, num_iterations))
    for _ in range(num_iterations):
        x = M @ x % den
        q = x[:n**2].reshape(n, n)
        q_avg[:,:,_] = q
    return q_avg

def run_simulation(n, k, g, Q0, P0,den, delta, num_iterations, num_runs,directory,showvals,coupling_range):
    
    C = generate_matrix_C_2D(n, k, g,coupling_range)
    M = np.block([[np.eye(n**2), C], [C, np.eye(n**2) + C @ C]])
    q_avg_over_runs = np.zeros((n, n, num_iterations))
    for run in range(num_runs):
        x = generate_initial_condition(n, Q0, P0, delta)
        q_avg = iterate(M, x,den, num_iterations, n)
        q_avg_over_runs += (q_avg - q_avg_over_runs) / (run + 1)
    for i in range(num_iterations):
        plt.close()
        fig, (ax1) = plt.subplots(nrows=1)
        fig.set_size_inches(16, 16, forward=True)
        sns.heatmap(q_avg_over_runs[:,:,i]/den,ax=ax1, annot=showvals, cmap="turbo", linewidths=0.001, cbar=True, vmin=0, vmax=1, cbar_kws={'label': 'Average position value q within cell'})
        #plt.title(f'Average q at iteration {i+1}')
        filename = os.path.join(directory, "{:04d}.png".format(i + 1))
        
        plt.tight_layout()
        fig.savefig(filename)
        if i==0:
            filename = os.path.join(directory, "{:04d}.png".format(i))
            fig.savefig(filename)
        plt.close()
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()
        fig.clf()
        #plt.show()
def Generate_Plot(parameters,para2):
    directory = "./images/"+genname(parameters)
    if os.path.exists(directory):
        
        return directory
    else:
        os.makedirs(directory)
    tuple_list = []

    for dictionary in para2:
        new_dict = {k: v.replace("\n", "") if isinstance(v, str) else v for k, v in dictionary.items()}
        for item in new_dict.items():
            tuple_list.append(item)
    par = parameters
    n = int( par[4])
    coupling_range=int( par[5])
    k = float(par[6])
    g = float(par[7])
    Q0 = float(par[9])
    P0 = float(par[10])
    den=float(par[8])
    delta = 10**int(par[11])#0#0.1
    num_iterations = int(par[3])
    num_runs = int(par[12])
    showvals=par[14]
    run_simulation(n, k, g, Q0, P0,den, delta, num_iterations, num_runs,directory,showvals,coupling_range)
    print(directory)
    return directory





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


types=[0,1,1,2,2,2,2,2,2,2,2,2,2,2,0,3]
class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.directory=''
        self.data_queue = queue.Queue()
        self.master.config(height=1000)
        self.grid(sticky='news')
        self.plot_lock = threading.Lock()
        self.current_time = 0
        self.images=None 
        self.delay = 1.0  # In seconds
        self.zoom = 1.0
        self.loop = tk.BooleanVar()
        self.loop.set(False)  # Set to False initially
        start = time.time()
        end = time.time()
        print("Time taken to load images: ", end - start, "seconds")
        self.playing = False

        # Initialize the canvas attribute
        self.canvas = tk.Canvas(self.master)
        #self.canvas.pack()
        # Create an image object on the canvas and get its ID
        self.update_image = self.canvas.create_image(0, 0, anchor='nw')

        self.default_values = [True, '','', 10,29, 1,1, 1,10, 1,0, -999,1,1, True,'plot' ]
        self.create_widgets()
        sys.stdout = StdoutRedirect(self.console_output)
    def clear_table(self):
        # Destroy each widget in each row
        for row in self.rows:
            for entry in row:
                entry.destroy()

        # Clear the rows list
        self.rows.clear()
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
    def create_widgets(self):
        # Headers for each column
        self.headers = ['Active','Q_0 Pertubation Type','P_0 Pertubation Type','Mappings (m)', 
                        'Number of Maps (n)', 'Coupling Range (l)', 'K-Fib (K)', 'Coupling Strength (G)', 
                        'Denominator', 'Q0', 'P0', 'Delta exponent', '# Samples','Bin Size', 'Show values on Graph', 'Plot Name']

        self.table_frame = tk.Frame(self)
        self.table_frame.grid(row=0, column=0, sticky='news')
        self.button_frame = tk.Frame(self.table_frame)
        self.button_frame.grid(row=2, column=0,sticky='news')

        self.fig_frame = tk.Frame(self)
        self.fig_frame.grid(row=0, column=1, sticky='news')

        self.table_canvas = tk.Canvas(self.table_frame)
        self.table_scrollbar_y = tk.Scrollbar(self.table_frame, orient='vertical', command=self.table_canvas.yview)
        self.table_scrollbar_x = tk.Scrollbar(self.table_frame, orient='horizontal', command=self.table_canvas.xview)

        self.scrollable_table_frame = ttk.Frame(self.table_canvas)
        self.scrollable_table_frame.bind('<Configure>', lambda e: self.table_canvas.configure(scrollregion=self.table_canvas.bbox('all')))
        self.table_canvas.create_window((0, 0), window=self.scrollable_table_frame, anchor='nw')
        self.table_canvas.configure(yscrollcommand=self.table_scrollbar_y.set, xscrollcommand=self.table_scrollbar_x.set,height=620)
        self.table_canvas.grid(row=0, column=0, sticky='news')
        self.table_scrollbar_y.grid(row=0, column=1, sticky='ns')
        self.table_scrollbar_x.grid(row=1, column=0, sticky='ew')

        self.fig_canvas = tk.Canvas(self.fig_frame)
        self.fig_scrollbar_y = tk.Scrollbar(self.fig_frame, orient='vertical', command=self.fig_canvas.yview)
        self.fig_scrollbar_x = tk.Scrollbar(self.fig_frame, orient='horizontal', command=self.fig_canvas.xview)

        self.scrollable_fig_frame = ttk.Frame(self.fig_canvas)
        self.scrollable_fig_frame.bind('<Configure>', lambda e: self.fig_canvas.configure(scrollregion=self.fig_canvas.bbox('all')))
        self.fig_canvas.create_window((0, 0), window=self.scrollable_fig_frame, anchor='nw')
        self.fig_canvas.configure(yscrollcommand=self.fig_scrollbar_y.set, xscrollcommand=self.fig_scrollbar_x.set,height=750,width=800)
        self.fig_canvas.grid(row=0, column=0, sticky='news')
        self.fig_scrollbar_y.grid(row=0, column=1, sticky='ns')
        self.fig_scrollbar_x.grid(row=1, column=0, sticky='ew')

        self.button_and_image_frame = tk.Frame(self.fig_frame)
        self.button_and_image_frame.grid(row=0, column=0, sticky='sw')

        self.image_frame = tk.Frame(self.button_and_image_frame)
        self.image_frame.grid(row=1, column=0)

        self.zoom_button_frame = tk.Frame(self.button_and_image_frame)
        self.zoom_button_frame.grid(row=0, column=0)


        # Headers
        for i, header in enumerate(self.headers):
            tk.Label(self.scrollable_table_frame, text=header).grid(row=i, column=0)
        self.run_all_button = tk.Button(self.scrollable_table_frame, text="Run All", command=self.run_all).grid(row=16, column=0)

        tk.Label(self.scrollable_table_frame, text='').grid(row=20, column=0)
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
    def create_vid_widgets(self):

        self.time_var = tk.StringVar()
        self.time_var.set("Time: 0")

        self.time_label = tk.Label(self.zoom_button_frame, textvariable=self.time_var)
        self.time_label.grid(row=0, column=0)

        self.next_button = tk.Button(self.zoom_button_frame, text="Next", command=self.show_next)
        self.next_button.grid(row=0, column=1)

        self.prev_button = tk.Button(self.zoom_button_frame, text="Previous", command=self.show_prev)
        self.prev_button.grid(row=0, column=2)

        self.loop_button = tk.Checkbutton(self.zoom_button_frame, text="Loop", variable=self.loop)
        self.loop_button.grid(row=0, column=3)

        self.play_button = tk.Button(self.zoom_button_frame, text="Play", command=self.start_auto_play)
        self.play_button.grid(row=0, column=4)

        self.stop_button = tk.Button(self.zoom_button_frame, text="Stop", command=self.stop_auto_play)
        self.stop_button.grid(row=0, column=5)

        self.speed_up_button = tk.Button(self.zoom_button_frame, text="Speed Up", command=self.speed_up)
        self.speed_up_button.grid(row=0, column=6)

        self.slow_down_button = tk.Button(self.zoom_button_frame, text="Slow Down", command=self.slow_down)
        self.slow_down_button.grid(row=0, column=7)

       # self.zoom_in_button = tk.Button(self.zoom_button_frame, text="Zoom In", command=self.zoom_in)
       # self.zoom_in_button.grid(row=0, column=10)

        #self.zoom_out_button = tk.Button(self.zoom_button_frame, text="Zoom Out", command=self.zoom_out)
        #self.zoom_out_button.grid(row=0, column=11)

        self.slider = tk.Scale(self.zoom_button_frame, from_=0, to=len(self.images)-1, orient=tk.HORIZONTAL, command=self.update_time)
        self.slider.grid(row=0, column=8)
        
        
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
            if i==1 or i==2 or i==13 or i==15:
                entry.config(state='disabled')
            row.append(entry)
        run_button = tk.Button(self.scrollable_table_frame)
        run_button["text"] = "Run"
        run_button["command"] = lambda: self.run_script(row)
        run_button.grid(row=len(self.headers), column=len(self.rows) + 1)
        self.rows.append(row)
    def load_images(self, folder):
        image_files = sorted([f for f in os.listdir(folder) if f.endswith('.png')])
        images = [ImageTk.PhotoImage(Image.open(os.path.join(folder, img)).resize((800, 700))) for img in image_files]
        return images

    def zoom_in(self):
        self.zoom *= 1.2
        #self.canvas.scale("all", 0, 0, 1.1, 1.1)
        #self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def zoom_out(self):
        self.zoom /= 1.2
        #self.canvas.scale("all", 0, 0, 0.9, 0.9)
       # self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def show_next(self):
        self.current_time += 1
        self.check_time_bounds()
        self.draw_figure()

    def show_prev(self):
        self.current_time -= 1
        self.check_time_bounds()
        self.draw_figure()

    def start_auto_play(self):
        self.playing = True
        try:
            if self.thread.is_alive():
                return
        except:
            1
        self.thread = threading.Thread(target=self.auto_play)
        self.thread.start()

    def stop_auto_play(self):
        self.playing = False

    def auto_play(self):
        self.current_time = 0
        while self.current_time < len(self.images) and self.playing:
            self.draw_figure()
            self.canvas.update()
            time.sleep(self.delay)
            self.current_time += 1
            if self.current_time >= len(self.images) and self.loop.get():
                self.current_time = 0

    def speed_up(self):
        self.delay /= 2

    def slow_down(self):
        self.delay *= 2

    def update_time(self, value):
        self.current_time = int(value)
        self.draw_figure()

    def draw_figure(self):
        #img=self.images[self.current_time].resize((int(600 * self.zoom_level), int(750 * self.zoom_level)))
        #img = self.original_image.resize((int(600 * self.zoom_level), int(750 * self.zoom_level)))
        #photo = ImageTk.PhotoImage(img)

        #self.fig_canvas.delete('all')
        #self.fig_canvas.config(scrollregion=(0, 0, img.width, img.height))
        self.fig_canvas.create_image(0, 0, image=self.images[self.current_time], anchor='nw') 

##        self.fig_canvas.image = photo
##        self.fig_canvas.itemconfig(image=self.images[self.current_time])
        self.time_var.set("Time: " + str(self.current_time))
        self.slider.set(self.current_time)

    def check_time_bounds(self):
        if self.current_time >= len(self.images):  # We reached the end of the time dimension
            self.current_time = len(self.images) - 1  # Keep it at the last index
        if self.current_time < 0:  # We reached the beginning of the time dimension
            self.current_time = 0  # Keep it at the first index
##    def generate_plot(self, data):
##        # Unpack the data
##
##        try:
##            (values_box, plot1, show_values, show_cbar, n, filepath) = data
##        except:
##            return data
##        plt.close()
##
##        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [5, 1]}, figsize=(8, 10))
##
##        sns.heatmap(plot1,annot=show_values,cmap="turbo", linewidths=0, ax=ax1, cbar=show_cbar,vmin=0, vmax=1,cbar_kws={'label': '(Average) position value q within cell'})
##
##        ax1.set_xlabel("$m^{th}$ mapping",size=16)
##        ax1.set_ylabel("$i^{th}$ cell in the chain",size=16)
##        ax1.set_ylim(1,n+1)
##        ax2.text(0.5, 0.5, values_box, ha='center', va='center', wrap=True,size=11)
##        ax2.axis('off')
##        plt.tight_layout()
##
##        fig.savefig(filepath,dpi=256)
##        return filepath

##    def display_plot(self, fig):
##        for widget in self.image_frame.winfo_children():
##            widget.destroy()
##        img = Image.open(fig)
##        self.original_image = img
##        self.zoom_level = 1
##        self.master.after(1, self.update_image)
##        print("Done")
##        return
##    def check_queue(self):
##        try:
##            plot_data = self.data_queue.get_nowait()  # Non-blocking get
##            if plot_data is not None:
##                self.display_plot(self.generate_plot(plot_data))
##            self.master.after(100, self.check_queue)  # Check again after 100 ms
##        except queue.Empty:
##            # If the queue was empty, check again after 100 ms
##            self.master.after(100, self.check_queue)
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
        self.directory=Generate_Plot(row_data, data)
        #if self.images==None:
        self.images = self.load_images(self.directory)
        self.create_vid_widgets()
        #thread = threading.Thread(target=self.generate_plot_thread, args=(row_data,data))
        #thread.start()
        # Put None as a signal for end of processing
        #self.data_queue.put(None)  # Signal for end of processing
        #self.check_queue()  # Start polling the queue

        #with self.plot_lock:
           # plot_data = Generate_Plot(row_data, data)
            #self.data_queue.put(plot_data)

##    def update_image(self):
##        if not self.fig_canvas.winfo_exists():
##            return
##
##        img = self.original_image.resize((int(600 * self.zoom_level), int(750 * self.zoom_level)))
##        photo = ImageTk.PhotoImage(img)
##
##        self.fig_canvas.delete('all')
##        self.fig_canvas.config(scrollregion=(0, 0, img.width, img.height))
##        self.fig_canvas.create_image(0, 0, image=photo, anchor='nw') 
##
##        self.fig_canvas.image = photo

    def run_all(self):
        for row in self.rows:
            self.run_script(row)
        # Signal that there's no more data

import ctypes
 
ctypes.windll.shcore.SetProcessDpiAwareness(1)
root = tk.Tk()
root.geometry("1300x800")
root.title("1D Chain Propegation")
app = Application(master=root)
app.mainloop()



