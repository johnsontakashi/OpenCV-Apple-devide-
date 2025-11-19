#!/usr/bin/env python3
"""
Graphical User Interface for Fruit Label Placement Analysis

Simple GUI to run the pipeline without command line arguments.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys
import subprocess
import json
from pathlib import Path

class FruitAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fruit Label Placement Analysis")
        self.root.geometry("800x600")
        
        # Variables
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar(value="output")
        self.config_file = tk.StringVar(value="config.yaml")
        
        self.setup_ui()
        
    def setup_ui(self):
        """Create the GUI layout."""
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="🍎 Fruit Label Placement Analysis", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Input directory selection
        ttk.Label(main_frame, text="Input Directory (RGB-D Images):").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.input_dir, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.select_input_dir).grid(row=1, column=2)
        
        # Output directory selection
        ttk.Label(main_frame, text="Output Directory:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_dir, width=50).grid(row=2, column=1, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.select_output_dir).grid(row=2, column=2)
        
        # Config file selection
        ttk.Label(main_frame, text="Configuration File:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.config_file, width=50).grid(row=3, column=1, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.select_config_file).grid(row=3, column=2)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=3, pady=20)
        
        # Run analysis button
        self.run_button = ttk.Button(button_frame, text="🚀 Run Analysis", 
                                    command=self.run_analysis, style="Accent.TButton")
        self.run_button.pack(side=tk.LEFT, padx=5)
        
        # Use test data button
        ttk.Button(button_frame, text="📊 Use Test Data", 
                  command=self.use_test_data).pack(side=tk.LEFT, padx=5)
        
        # View results button
        self.view_button = ttk.Button(button_frame, text="📁 View Results", 
                                     command=self.view_results, state="disabled")
        self.view_button.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready to process images...")
        self.status_label.grid(row=6, column=0, columnspan=3, pady=5)
        
        # Log output
        ttk.Label(main_frame, text="Processing Log:").grid(row=7, column=0, sticky=tk.W, pady=(20, 5))
        
        self.log_text = scrolledtext.ScrolledText(main_frame, height=15, width=80)
        self.log_text.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(8, weight=1)
        
        # Check if test data exists
        if os.path.exists("test_data"):
            self.input_dir.set("test_data")
            self.log_message("✓ Test data found and loaded")
        else:
            self.log_message("ℹ️ Select input directory with RGB-D image pairs")
    
    def select_input_dir(self):
        """Select input directory containing RGB-D images."""
        directory = filedialog.askdirectory(title="Select Input Directory with RGB-D Images")
        if directory:
            self.input_dir.set(directory)
            self.log_message(f"✓ Input directory selected: {directory}")
            self.check_image_pairs(directory)
    
    def select_output_dir(self):
        """Select output directory."""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir.set(directory)
            self.log_message(f"✓ Output directory selected: {directory}")
    
    def select_config_file(self):
        """Select configuration file."""
        filename = filedialog.askopenfilename(
            title="Select Configuration File",
            filetypes=[("YAML files", "*.yaml"), ("YAML files", "*.yml"), ("All files", "*.*")]
        )
        if filename:
            self.config_file.set(filename)
            self.log_message(f"✓ Configuration file selected: {filename}")
    
    def check_image_pairs(self, directory):
        """Check and report image pairs in directory."""
        try:
            image_files = []
            depth_files = []
            
            for file_path in Path(directory).iterdir():
                if file_path.is_file():
                    name = file_path.name.lower()
                    if any(ext in name for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
                        if 'depth' in name:
                            depth_files.append(str(file_path))
                        else:
                            image_files.append(str(file_path))
            
            self.log_message(f"📊 Found {len(image_files)} color images and {len(depth_files)} depth images")
            
            if len(image_files) == 0:
                self.log_message("⚠️ No color images found!")
            elif len(depth_files) == 0:
                self.log_message("⚠️ No depth images found! (should contain 'depth' in filename)")
            else:
                self.log_message("✓ Ready to process!")
                
        except Exception as e:
            self.log_message(f"❌ Error checking directory: {e}")
    
    def use_test_data(self):
        """Use the test data directory."""
        if os.path.exists("test_data"):
            self.input_dir.set("test_data")
            self.log_message("✓ Using test data (synthetic fruit images)")
        else:
            self.log_message("❌ Test data not found. Please create test data first.")
            if messagebox.askyesno("Create Test Data", 
                                 "Test data not found. Would you like to create synthetic test data?"):
                self.create_test_data()
    
    def create_test_data(self):
        """Create synthetic test data."""
        try:
            self.log_message("🔄 Creating synthetic test data...")
            result = subprocess.run([sys.executable, "create_test_data.py"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                self.log_message("✓ Test data created successfully!")
                self.input_dir.set("test_data")
            else:
                self.log_message(f"❌ Error creating test data: {result.stderr}")
                
        except Exception as e:
            self.log_message(f"❌ Error creating test data: {e}")
    
    def run_analysis(self):
        """Run the fruit analysis pipeline."""
        if not self.input_dir.get():
            messagebox.showerror("Error", "Please select an input directory!")
            return
            
        if not os.path.exists(self.config_file.get()):
            messagebox.showerror("Error", f"Configuration file not found: {self.config_file.get()}")
            return
        
        # Start analysis in separate thread
        self.run_button.config(state="disabled")
        self.progress.start()
        self.status_label.config(text="🔄 Running analysis...")
        self.log_text.delete(1.0, tk.END)
        
        thread = threading.Thread(target=self._run_analysis_thread)
        thread.daemon = True
        thread.start()
    
    def _run_analysis_thread(self):
        """Run analysis in background thread."""
        try:
            # Build command
            cmd = [
                sys.executable, "main.py",
                "--input", self.input_dir.get(),
                "--config", self.config_file.get(),
                "--output", self.output_dir.get()
            ]
            
            self.log_message(f"🚀 Starting analysis: {' '.join(cmd)}")
            
            # Run the pipeline
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Read output line by line
            for line in process.stdout:
                line = line.strip()
                if line:
                    # Update GUI in main thread
                    self.root.after(0, lambda l=line: self.log_message(l))
            
            # Wait for completion
            process.wait()
            
            if process.returncode == 0:
                self.root.after(0, self._analysis_completed)
            else:
                self.root.after(0, lambda: self._analysis_failed("Pipeline failed"))
                
        except Exception as e:
            self.root.after(0, lambda: self._analysis_failed(str(e)))
    
    def _analysis_completed(self):
        """Handle successful analysis completion."""
        self.progress.stop()
        self.run_button.config(state="normal")
        self.view_button.config(state="normal")
        self.status_label.config(text="✅ Analysis completed successfully!")
        
        # Show results summary
        try:
            results_file = os.path.join(self.output_dir.get(), "results.json")
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                summary = results.get('processing_summary', {})
                msg = f"""Analysis Complete! 🎉

📊 Results Summary:
• Images processed: {summary.get('successful_images', 0)}/{summary.get('total_images', 0)}
• Fruits detected: {summary.get('total_fruits', 0)}
• Label candidates found: {summary.get('total_candidates', 0)}

📁 Output saved to: {self.output_dir.get()}

Would you like to view the results?"""
                
                if messagebox.askyesno("Analysis Complete", msg):
                    self.view_results()
                    
        except Exception as e:
            self.log_message(f"⚠️ Could not read results summary: {e}")
    
    def _analysis_failed(self, error_msg):
        """Handle analysis failure."""
        self.progress.stop()
        self.run_button.config(state="normal")
        self.status_label.config(text="❌ Analysis failed")
        self.log_message(f"❌ Analysis failed: {error_msg}")
        messagebox.showerror("Analysis Failed", f"The analysis failed:\n\n{error_msg}")
    
    def view_results(self):
        """Open the results directory."""
        output_path = self.output_dir.get()
        
        if not os.path.exists(output_path):
            messagebox.showerror("Error", f"Output directory not found: {output_path}")
            return
        
        try:
            # Open file manager to results directory
            if sys.platform == "win32":
                os.startfile(output_path)
            elif sys.platform == "darwin":  # macOS
                subprocess.run(["open", output_path])
            else:  # Linux
                subprocess.run(["xdg-open", output_path])
                
            self.log_message(f"📁 Opened results directory: {output_path}")
            
        except Exception as e:
            self.log_message(f"❌ Could not open results directory: {e}")
            messagebox.showinfo("Results Location", f"Results saved to:\n{output_path}")
    
    def log_message(self, message):
        """Add message to log output."""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()


def main():
    """Run the GUI application."""
    # Check if we're in the right directory
    if not os.path.exists("config.yaml"):
        print("Error: config.yaml not found. Please run this GUI from the project directory.")
        return
    
    root = tk.Tk()
    app = FruitAnalysisGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()