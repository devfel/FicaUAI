# ../gui.py

import tkinter as tk
import numpy as np
import pandas as pd
from tkinter import ttk, filedialog
import main
import tk_tools
import tkinter.font as tkFont
from sklearn.metrics import confusion_matrix

NUM_EXECUTIONS = 5  # Number of times to run the model with different random seeds trying to find best log_loss


class DataScienceGUI:
    def __init__(self, root):
        self.root = root
        root.minsize(400, 680)  # sets the minimum size of the window
        root.maxsize(400, 680)  # sets the minimum size of the window
        root.geometry("400x680")  # Width x Height

        self.root.title("ANN MLP Data Science - Evasão")
        self.data_model = main.DataModel()  # Create an instance of DataModel

        self.accuracy_var = tk.StringVar(value="N/A")
        self.loss_var = tk.StringVar(value="N/A")

        # Create a Style object
        style = ttk.Style()

        # # Choose a theme
        # style.theme_use(
        #     "clam"
        # )  # Try 'clam', 'alt', 'default', 'classic', 'vista', 'xpnative', 'default'

        # # Configure the style of the tab
        style.configure(
            "TNotebook.Tab",
            foreground="black",
            background="white",
            padding=[5, 2],
            # font=("Helvetica", 10, "bold"),
            font=("Helvetica", 10, "bold"),
        )

        self.tab_control = ttk.Notebook(root)

        self.n_hidden_layers_var = tk.StringVar(value="N/A")
        self.total_hidden_neurons_var = tk.StringVar(value="N/A")

        # Tab 1: File selection
        self.tab1 = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab1, text="Ler Arquivo")
        self.setup_tab1()

        # Tab 2: Model Configuration (Initially disabled)
        self.tab2 = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab2, text="Variáveis")
        self.tab_control.tab(1, state="disabled")  # Disable tab 2 initially

        # Tab 3: Results (Initially disabled)
        self.tab3 = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab3, text="Resultados")
        self.tab_control.tab(2, state="disabled")  # Disable tab 3 initially

        # Tab 4: New Student Prediction (Initially disabled)
        self.tab4 = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab4, text="Termômetro")
        self.tab_control.tab(3, state="disabled")  # Disable tab 4 initially

        # Tab 5: List Students (Initially disabled)
        self.tab5 = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab5, text="Lista")
        self.tab_control.tab(4, state="disabled")  # Disable tab 5 initially

        self.tab_control.pack(expand=1, fill="both")

        # You can define min_prob and max_prob here or elsewhere in your class
        self.min_prob = 0
        self.max_prob = 0

        self.tab_control.pack(expand=1, fill="both")

    def setup_tab1(self):

        # Elements for file selection and loading
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(self.tab1, textvariable=self.file_path_var, width=50)
        file_entry.grid(column=0, row=0, padx=5, pady=10)
        file_button = ttk.Button(self.tab1, text="Procurar", command=self.browse_file)
        file_button.grid(column=1, row=0, padx=5, pady=10)

        # Adding Business Rules Information
        rules_label = ttk.Label(
            self.tab1,
            text="=== INSTRUÇÕES: ===\n\n"
            "1) O arquivo DEVE ser um arquivo EXCEL, .xls ou .xlsx.\n"
            "2) A primeira linha do arquivo DEVE conter os nomes das variáveis.\n"
            "3) Adicione uma coluna ID com identificadores únicos do tipo INTEIRO.\n"
            "4) As variáveis utilizadas NÃO podem conter dados faltantes ou vazios.\n"
            "5) As variáveis utilizadas DEVEM ser numéricas, mesmo categorias.\n"
            "6) A variável dependente DEVE ser 1 para EVADIU e 0 NÃO EVADIU.\n\n"
            "=== EXEMPLO DE ARQUIVO VÁLIDO: ===",
            justify=tk.LEFT,
            anchor="w",
        )
        rules_label.grid(column=0, row=2, columnspan=2, padx=5, pady=10, sticky="w")

        # Frame to hold the Treeview for the example of a valid file
        example_frame = ttk.Frame(self.tab1)
        example_frame.grid(column=0, row=3, columnspan=2, padx=10, pady=0, sticky="ew")

        # Create a Treeview widget within the frame
        example_tree = ttk.Treeview(
            example_frame,
            columns=("ID", "Evasão", "Notas", "Falta(%)", "ENEM"),
            show="headings",
        )
        example_tree.heading("ID", text="ID")
        example_tree.heading("Evasão", text="Evasão")
        example_tree.heading("Notas", text="Notas")
        example_tree.heading("Falta(%)", text="Falta(%)")
        example_tree.heading("ENEM", text="ENEM")

        example_tree.column("ID", anchor="center", width=5)
        example_tree.column("Evasão", anchor="center", width=5)
        example_tree.column("Notas", anchor="center", width=5)
        example_tree.column("Falta(%)", anchor="center", width=5)
        example_tree.column("ENEM", anchor="center", width=5)

        # Inserting the example data
        example_data = [
            ("1350", "1", "75", "12", "650"),
            ("1351", "0", "80", "15", "645"),
            ("1352", "0", "89", "7", "716"),
            ("1353", "1", "42", "20", "597"),
            ("1354", "0", "50", "23", "614"),
            ("1355", "1", "20", "11", "632"),
        ]

        for item in example_data:
            example_tree.insert("", "end", values=item)

        # Configure the Treeview widget to expand and fill the space
        example_tree.pack(side="left", expand=True, fill="both")

    def browse_file(self):
        file_path = filedialog.askopenfilename()
        self.file_path_var.set(file_path)
        if file_path:
            # Check if the file is an Excel file
            if file_path.endswith(".xls") or file_path.endswith(
                ".xlsx"
            ):  # check if the file is an Excel file
                self.data_model.load_data(file_path)  # Load the file using DataModel
                self.setup_tab2()  # Automatically refresh variables in Tab 2
                self.tab_control.tab(1, state="normal")  # Enable Tab 2

                # Check if the widget exists already
                if hasattr(self, "file_loaded_label"):
                    self.file_loaded_label.destroy()  # Remove old label if exists

                # Add a ttk.Label to show confirmation message
                self.file_loaded_label = ttk.Label(
                    self.tab1,
                    text=f"Arquivo carregado com sucesso.",  #'{file_path}' to add path/file name.
                )
                self.file_loaded_label.grid(column=0, row=1, columnspan=2, pady=10)
                self.tab_control.tab(2, state="disabled")  # Disable Tab 3
                self.tab_control.tab(3, state="disabled")  # Disable Tab 4
                self.tab_control.tab(4, state="disabled")  # Disable Tab 5
            else:
                # add a label to show the error message
                if hasattr(self, "file_loaded_label"):
                    self.file_loaded_label.destroy()  # Remove old label

                self.file_loaded_label = ttk.Label(
                    self.tab1,
                    text="Tipo de arquivo inválido. Selecione um arquivo .xls ou .xlsx.",
                )
                self.file_loaded_label.grid(column=0, row=1, columnspan=2, pady=10)
                self.tab_control.tab(1, state="disabled")  # Disable Tab 2
                self.tab_control.tab(2, state="disabled")  # Disable Tab 3
                self.tab_control.tab(3, state="disabled")  # Disable Tab 4
                self.tab_control.tab(4, state="disabled")  # Disable Tab 5

        else:
            # if no file was selected
            if hasattr(self, "file_loaded_label"):
                self.file_loaded_label.destroy()  # remove old label

            self.file_loaded_label = ttk.Label(
                self.tab1, text="Nenhum arquivo foi selecionado."
            )
            self.file_loaded_label.grid(column=0, row=1, columnspan=2, pady=10)
            self.tab_control.tab(1, state="disabled")  # Disable Tab 2
            self.tab_control.tab(2, state="disabled")  # Disable Tab 3
            self.tab_control.tab(3, state="disabled")  # Disable Tab 4
            self.tab_control.tab(4, state="disabled")  # Disable Tab 5

    def setup_tab2(self):

        # Clear existing widgets in tab2 to refresh it dynamically
        for widget in self.tab2.winfo_children():
            widget.destroy()

        # Check if data is loaded
        if self.data_model.data is not None:
            options = self.data_model.data.columns.tolist()

            # Dependent Variable Selection
            self.dependent_var = tk.StringVar()
            dependent_label = ttk.Label(self.tab2, text="Variável Dependente:")
            dependent_label.grid(column=0, row=0, padx=10, pady=(15, 5), sticky="w")
            dependent_dropdown = ttk.Combobox(
                self.tab2,
                textvariable=self.dependent_var,
                values=options,
                state="readonly",
                width=30,
            )
            dependent_dropdown.grid(column=1, row=0, padx=10, pady=(15, 5), sticky="w")

            # Insert Separator Here
            separator = ttk.Separator(self.tab2, orient="horizontal")
            separator.grid(column=0, row=1, columnspan=2, padx=10, pady=10, sticky="ew")

            # Covariates Selection
            covariates_label = ttk.Label(self.tab2, text="Covariáveis:")
            covariates_label.grid(column=0, row=2, padx=10, pady=5, sticky="w")

            # Creating a frame for the Listbox and Scrollbar for better layout control
            covariates_frame = tk.Frame(self.tab2)
            covariates_frame.grid(column=1, row=2, padx=10, pady=5, sticky="w")

            self.covariates_listbox = tk.Listbox(
                covariates_frame,
                selectmode="multiple",
                exportselection=0,
                height=20,
                width=30,
            )
            self.covariates_listbox.pack(side="left", fill="y")

            scrollbar = tk.Scrollbar(covariates_frame, orient="vertical")
            scrollbar.pack(side="right", fill="y")

            # Link scrollbar to listbox
            self.covariates_listbox.config(yscrollcommand=scrollbar.set)
            scrollbar.config(command=self.covariates_listbox.yview)

            for option in options:
                self.covariates_listbox.insert(tk.END, option)

            # Submit Button to Apply Selection
            submit_button = ttk.Button(
                self.tab2, text="Aplicar Seleção", command=self.apply_selection
            )
            submit_button.grid(column=1, row=3, padx=10, pady=5)

    def apply_selection(self):
        self.tab_control.tab(3, state="disabled")  # Disable Tab 4
        self.tab_control.tab(4, state="disabled")  # Disable Tab 5
        # TODO Reset Results from the Results tab
        # Reset Results in the Results tab (Tab 3)
        self.accuracy_var.set("N/A")
        self.loss_var.set("N/A")
        self.n_hidden_layers_var.set("N/A")
        self.total_hidden_neurons_var.set("N/A")

        # Get selected dependent variable
        dependent = self.dependent_var.get()

        # Get selected covariates
        selected_indices = self.covariates_listbox.curselection()
        covariates = [self.covariates_listbox.get(i) for i in selected_indices]

        print(f"Selected Variável Dependente: {dependent}")
        print(f"Selected Covariáveis: {covariates}")

        self.selected_covariates = [
            self.covariates_listbox.get(i) for i in selected_indices
        ]

        # Update the DataModel instance with the selected covariates
        self.data_model.selected_covariates = self.selected_covariates

        # Proceed with model configuration using selected dependent and covariates
        # Enable tab 3 once the model is configured and trained
        self.tab_control.tab(2, state="normal")  # Enable tab 3
        self.setup_tab3()  # Initialize Tab 3 contents
        self.setup_tab4()

        # Check if the widget exists already
        if hasattr(self, "variables_loaded_label"):
            self.variables_loaded_label.destroy()  # Remove old label if exists

        # Add a ttk.Label to show confirmation message
        self.variables_loaded_label = ttk.Label(
            self.tab2, text="Variáveis Carregadas Corretamente"
        )
        self.variables_loaded_label.grid(
            column=0, row=4, columnspan=2, padx=10, pady=10
        )

    def setup_tab3(self):

        fontB = tkFont.Font(family="Helvetica", size=10, weight="bold")

        # Frame for the buttons
        buttons_frame = ttk.Frame(self.tab3)
        buttons_frame.grid(
            column=0, row=0, padx=10, pady=5, sticky="ew", columnspan=2
        )  # Spanning 2 columns
        self.tab3.columnconfigure(0, weight=1)
        self.tab3.columnconfigure(1, weight=1)

        # Button to Run Model
        run_model_button = ttk.Button(
            buttons_frame,
            text="Executar Rede Neural",
            command=lambda: [
                self.run_model(),
                # run_model_button.config(state=tk.DISABLED), # Debug
            ],
        )
        run_model_button.grid(column=0, row=0, padx=5, pady=5, sticky="w")

        # Button to Save Results
        self.save_button = ttk.Button(
            buttons_frame,
            text="Criar Excel com Probabilidades",
            command=self.save_results,
            state="disabled",
        )
        self.save_button.grid(column=1, row=0, padx=5, pady=5, sticky="w")

        # Adding a LabelFrame for structured presentation
        results_frame = ttk.LabelFrame(
            self.tab3, text="Resultados do Modelo", padding="10"
        )
        results_frame.grid(
            column=0, row=1, padx=10, pady=10, sticky="ew", columnspan=2
        )  # Use columnspan=2 if needed

        # Ensure results_frame stretches across the window
        self.tab3.columnconfigure(0, weight=1)
        results_frame.columnconfigure(
            1, weight=1
        )  # Make sure results are properly aligned

        # Accuracy Display
        accuracy_label = ttk.Label(results_frame, text="Acurácia em Testes")
        accuracy_label.grid(column=0, row=0, padx=5, pady=5, sticky="w")
        accuracy_result = ttk.Label(
            results_frame, textvariable=self.accuracy_var, font=fontB
        )
        accuracy_result.grid(column=1, row=0, padx=5, pady=5, sticky="w")

        # Log Loss Display
        loss_label = ttk.Label(results_frame, text="Erro Cross-Entropy")
        loss_label.grid(column=0, row=1, padx=5, pady=5, sticky="w")
        loss_result = ttk.Label(results_frame, textvariable=self.loss_var, font=fontB)
        loss_result.grid(column=1, row=1, padx=5, pady=5, sticky="w")

        # Hidden Layers Display
        hidden_layers_label = ttk.Label(results_frame, text="Qtd. de Camadas Ocultas")
        hidden_layers_label.grid(column=2, row=0, padx=5, pady=5, sticky="w")
        hidden_layers_result = ttk.Label(
            results_frame, textvariable=self.n_hidden_layers_var, font=fontB
        )
        hidden_layers_result.grid(column=3, row=0, padx=5, pady=5, sticky="w")

        # Total Neurons Display
        total_neurons_label = ttk.Label(
            results_frame, text="Total de Neurônios Ocultos"
        )
        total_neurons_label.grid(column=2, row=1, padx=5, pady=5, sticky="w")
        total_neurons_result = ttk.Label(
            results_frame, textvariable=self.total_hidden_neurons_var, font=fontB
        )
        total_neurons_result.grid(column=3, row=1, padx=5, pady=5, sticky="w")

        # Initialize StringVar for confusion matrix results
        self.confusion_var = tk.StringVar(
            value="Os resultados da Matriz de Confusão irão aparecer aqui"
        )

        # Display for Confusion Matrix
        confusion_label = ttk.Label(
            self.tab3, text="Resultados - Matriz de Confusão:", font=fontB
        )
        confusion_label.grid(column=0, row=2, padx=10, pady=5, sticky="w")
        confusion_result = ttk.Label(
            self.tab3, textvariable=self.confusion_var, justify=tk.LEFT
        )
        confusion_result.grid(column=0, row=3, padx=10, pady=5, sticky="w")

        # Adjusting column configuration to ensure it stretches with the window size
        self.tab3.columnconfigure(0, weight=1)

    def run_model(self):
        # Ensure existing feature importances are cleared before recalculating
        self.data_model.feature_importances.clear()

        # Prepare data, train, and evaluate the model
        dependent = self.dependent_var.get()
        selected_indices = self.covariates_listbox.curselection()
        covariates = [self.covariates_listbox.get(i) for i in selected_indices]

        if dependent and covariates:
            self.data_model.prepare_data(x_columns=covariates, y_column=dependent)
            best_log_loss = float("inf")
            best_seed_value = None

            for _ in range(NUM_EXECUTIONS):
                # Change here for Random or Fixed Seed (for reproducibility)
                current_seed = np.random.randint(0, 2147483647)
                # current_seed = (
                #     1424222372  # Change here to a fixed seed for reproducibility
                # )
                X_train, X_test, y_train, y_test = self.data_model.split_data(
                    seed=current_seed
                )
                self.data_model.train_model(X_train, y_train)
                current_accuracy, current_log_loss = self.data_model.evaluate_model(
                    X_test, y_test
                )

                if current_log_loss < best_log_loss:
                    best_log_loss = current_log_loss
                    best_seed_value = current_seed

            # After finding the best seed, train the model again with the best seed
            if best_seed_value is not None:
                X_train, X_test, y_train, y_test = self.data_model.split_data(
                    seed=best_seed_value
                )
                self.data_model.train_model(X_train, y_train)
                melhor_acuracia, _ = self.data_model.evaluate_model(X_test, y_test)
                self.accuracy_var.set(f"{melhor_acuracia:.4f}")
                self.loss_var.set(f"{best_log_loss:.4f}")

                # Update the variables sizes used on the training
                self.n_hidden_layers_var.set(str(self.data_model.n_hidden_layers))
                self.total_hidden_neurons_var.set(
                    str(self.data_model.total_hidden_neurons)
                )
            else:
                print("No seeds found.")
        else:
            self.accuracy_var.set("N/A")
            self.loss_var.set("N/A")
            print("Error: Model configuration seems incomplete.")

        # After evaluating the model:
        predictions_train = self.data_model.model.predict(X_train)
        predictions_test = self.data_model.model.predict(X_test)
        y_combined_true = np.concatenate((y_train, y_test))
        predictions_combined = np.concatenate((predictions_train, predictions_test))

        # Generating confusion matrix results for display
        confusion_results = ""
        confusion_results += self.data_model.detailed_confusion(
            y_train, predictions_train, "Treinamento"
        )
        confusion_results += self.data_model.detailed_confusion(
            y_test, predictions_test, "Teste"
        )
        confusion_results += self.data_model.detailed_confusion(
            y_combined_true, predictions_combined, "Total (Treino + Teste)"
        )

        # Update the GUI with confusion matrix results
        self.confusion_var.set(
            confusion_results
        )  # Assuming you have a StringVar for this

        self.tab_control.tab(3, state="normal")
        self.tab_control.tab(4, state="normal")
        self.save_button["state"] = "normal"

        X_train_df = pd.DataFrame(X_train, columns=covariates)
        self.data_model.calculate_feature_importances(X_train_df, y_train, covariates)
        self.data_model.normalize_feature_importances()  # self.data_model.normalize_feature_importances() OR self.data_model.relative_feature_importances()
        self.setup_tab4()
        self.setup_tab5()

    def setup_tab4(self):

        # Clear existing widgets in tab4 to refresh it dynamically
        for widget in self.tab4.winfo_children():
            widget.destroy()

        self.tab4.columnconfigure(0, weight=1)  # Ensure the column in tab4 can expand
        self.bold_font = tkFont.Font(
            family="Helvetica", size=10, weight="bold"
        )  # Create a bold font for labels

        # Adding a LabelFrame for search functionality
        search_frame = ttk.LabelFrame(self.tab4, text="Buscar por ID", padding="10")
        search_frame.grid(
            column=0, row=0, padx=10, pady=10, sticky="nsew", columnspan=3
        )
        search_frame.columnconfigure(
            0, weight=1
        )  # Make sure entry field expands to fill the frame
        # No need to adjust weight for the button column (1), since we want it to remain its size

        # Search Field
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.grid(column=0, row=0, padx=5, pady=5, sticky="ew")

        # Search Button
        search_button = ttk.Button(
            search_frame, text="Buscar", command=self.fetch_data_by_id
        )
        search_button.grid(column=1, row=0, padx=5, pady=5)

        # Static descriptive text Label for "Aluno evadiu:"
        dep_var_static_label = ttk.Label(search_frame, text="Aluno evadiu:")
        dep_var_static_label.grid(column=2, row=0, padx=5, pady=5)

        # Dynamic content Label for "Sim" or "Não" or "N/A"
        self.dep_var_label_var = tk.StringVar(
            value="N/A"
        )  # Default display value for dynamic content
        dep_var_label = ttk.Label(search_frame, textvariable=self.dep_var_label_var)
        dep_var_label.grid(column=3, row=0, padx=5, pady=5)

        # to update a property (like self.data_model.feature_importances) with the calculated importances
        feature_importances = (
            self.data_model.feature_importances
        )  # Or fetch directly if needed

        # Create a Canvas for the input fields section
        input_fields_canvas = tk.Canvas(self.tab4)
        input_fields_scrollbar = ttk.Scrollbar(
            self.tab4, orient="vertical", command=input_fields_canvas.yview
        )
        input_fields_canvas.configure(yscrollcommand=input_fields_scrollbar.set)

        # Frame to hold the input fields inside the canvas
        input_fields_frame = ttk.Frame(input_fields_canvas)
        input_fields_canvas.create_window(
            (0, 0), window=input_fields_frame, anchor="nw"
        )

        input_fields_frame.bind(
            "<Configure>",
            lambda e: input_fields_canvas.configure(
                scrollregion=input_fields_canvas.bbox("all")
            ),
        )

        # Position the canvas and scrollbar in the tab4 layout
        input_fields_canvas.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        input_fields_scrollbar.grid(row=1, column=1, sticky="ns")

        self.tab4.rowconfigure(1, weight=1)
        self.tab4.columnconfigure(0, weight=1)

        # Add input fields to the input_fields_frame instead of self.tab4
        self.new_student_vars = {}
        row = 0  # Start from row 0 in the input fields frame
        for covariate in self.selected_covariates:
            importance_percentage = feature_importances.get(
                covariate, 0
            )  # Default to 0%
            # Truncate covariate name to 24 characters for display purpuses
            truncated_covariate = covariate[:24]
            label_text = f"{truncated_covariate} ({importance_percentage:.2f}%):"
            label = ttk.Label(input_fields_frame, text=label_text)
            label.grid(column=0, row=row, padx=10, pady=5, sticky="w")

            entry_var = tk.StringVar()
            entry = ttk.Entry(input_fields_frame, textvariable=entry_var)
            entry.grid(column=1, row=row, padx=10, pady=5, sticky="ew")

            self.new_student_vars[covariate] = entry_var
            row += 1  # Increment row for the next input field

        # Button to Calculate Dropout Probability
        calc_button = ttk.Button(
            self.tab4,
            text="Calcular Probabilidade de Evasão %",
            command=self.calculate_dropout_probability,
        )
        calc_button.grid(column=0, row=row, padx=10, pady=5, columnspan=2)
        row += 1  # Increment row for the probability label

        # Label to display probability
        self.probability_var = tk.StringVar(value="Probabilidade será exibida aqui.")
        probability_label = ttk.Label(
            self.tab4, textvariable=self.probability_var, font=self.bold_font
        )
        probability_label.grid(column=0, row=row, padx=10, pady=5, columnspan=2)

        row += 1  # Increment row again for the gauge

        # Adding a Gauge for Dropout Probability
        self.dropout_gauge = tk_tools.Gauge(
            self.tab4,
            height=200,
            width=390,
            max_value=100.0,
            min_value=0.0,
            yellow=30,
            red=70,
            unit="%",
            divisions=10,
            label="Probabilidade de Evasão",
            bg="gray70",
        )
        self.dropout_gauge.grid(
            row=row, column=0, padx=5, pady=10, sticky="news", columnspan=2
        )

        # Ensure the new layout adjusts dynamically
        self.tab4.rowconfigure(row, weight=1)

    def fetch_data_by_id(self):
        # Get the ID from the entry field
        search_id = self.search_var.get()

        try:
            # Ensure search_id is converted to the correct type (int in this case)
            search_id = int(search_id)

            # Find the row in the DataFrame where the ID matches
            row_data = self.data_model.data.loc[self.data_model.data["ID"] == search_id]

            # If no data is found for the ID, you can update your GUI to reflect that
            if row_data.empty:
                print("No data found for ID:", search_id)
                # Clear the fields or show a message in the GUI
                for covariate in self.new_student_vars:
                    self.new_student_vars[covariate].set("")
                self.probability_var.set("ID não encontrado.")

                # Directly set the gauge to 0
                self.dropout_gauge.set_value(0)

                # Update this line to reset the dependent variable label to "N/A"
                self.dep_var_label_var.set("N/A")
                return

            # Extract the row as a Series
            data_for_id = row_data.iloc[0]

            # Loop over the covariate entry fields and set their values based on the data
            for covariate in self.new_student_vars:
                # Check if covariate is in the DataFrame columns
                if covariate in data_for_id:
                    self.new_student_vars[covariate].set(data_for_id[covariate])
                else:
                    print(f"Covariate {covariate} not in the DataFrame")

            # Get the name of the selected dependent variable
            selected_dep_var_name = self.dependent_var.get()

            # Update the dependent variable label based on the data searched
            if selected_dep_var_name in data_for_id:
                dep_var_value = data_for_id[selected_dep_var_name]
                # Determine what to display based on the value
                self.dep_var_label_var.set("Sim" if dep_var_value == 1 else "Não")
            else:
                # If the dependent variable name is not in the row data, or no selection has been made
                self.dep_var_label_var.set("N/A")

            self.calculate_dropout_probability()  # Re-Calculate the dropout probability

        except ValueError:
            # Handle the case where the ID is not an integer
            print("ID should be an integer.")
            self.probability_var.set("ID deve ser um número inteiro.")
            self.dropout_gauge.set_value(0)  # Set gauge to 0 for invalid ID
        except Exception as e:
            # Handle other exceptions such as issues with data access
            print(f"An error occurred while searching ID: {e}")
            self.probability_var.set(f"Ocorreu um erro ao buscar: {e}")
            self.dropout_gauge.set_value(0)  # Set gauge to 0 for invalid ID

    def calculate_dropout_probability(self, student_data=None):
        if student_data is None:
            new_student_data = {
                covariate: [float(self.new_student_vars[covariate].get())]
                for covariate in self.selected_covariates
            }
            new_student_df = pd.DataFrame(new_student_data)
            probability = self.data_model.predict_new_student(new_student_df) * 100
            self.probability_var.set(f"Probabilidade de Evasão: {probability:.2f}%")

            # Ensure the probability is within the gauge's limits due to errors in gauge toolkit not handling too close to limit values.
            if probability > 97.5:
                gauge_probability = 97
            elif probability < 1:
                gauge_probability = 1
            else:
                gauge_probability = probability

            # Update the gauge with the new probability, rounded to one decimal place
            self.dropout_gauge.set_value(round(gauge_probability, 2))

        if student_data is not None:
            # Adjusting the creation of the DataFrame to ensure each value is in a list
            new_student_data = {
                covariate: [value] for covariate, value in student_data.items()
            }
            new_student_df = pd.DataFrame(new_student_data)
            probability = self.data_model.predict_new_student(new_student_df) * 100
            return probability

    def list_students_within_probability_range(self):
        # Convert input fields to float values
        try:
            self.min_prob = float(self.min_prob_input.get())
            self.max_prob = float(self.max_prob_input.get())
        except ValueError:
            # Handle the case where the input is not a valid float
            tk.messagebox.showerror(
                "Erro", "Insira valores numéricos válidos para as probabilidades."
            )
            return

        # Clear existing entries in the Treeview
        for i in self.students_tree.get_children():
            self.students_tree.delete(i)

        # Check if data is loaded
        if self.data_model.data is not None:
            try:
                for index, row in self.data_model.data.iterrows():
                    # Skip rows based on the checkbox state and dependent variable value
                    selected_dep_var_name = (
                        self.dependent_var.get()
                    )  # Get the name of the selected dependent variable
                    if (
                        self.show_non_dropout_var.get()
                        and row[selected_dep_var_name] != 0
                    ):
                        continue  # Skip this row

                    student_data = {
                        covariate: row[covariate]
                        for covariate in self.selected_covariates
                    }
                    # Convert the student's data into the format expected by your modified calculate_dropout_probability
                    probability = self.calculate_dropout_probability(
                        student_data
                    )  # Adjusted to accept data and return probability

                    if self.min_prob <= probability <= self.max_prob:
                        # Make sure 'ID' column is present
                        if "ID" in row:
                            self.students_tree.insert(
                                "",
                                tk.END,
                                values=(int(row["ID"]), f"{probability:.2f}"),
                            )
                        else:
                            raise KeyError(
                                "Está faltando a Coluna ID no arquivo carregado."
                            )

                    # After populating the Treeview, sort by "Probabilidade (%)" in descending order "Second Parameter = TRUE"
                    self.treeview_sort_column(
                        self.students_tree, "Probabilidade (%)", True
                    )
            except KeyError as e:
                tk.messagebox.showerror("Erro", f"Coluna não encontrada: {e}")
            except Exception as e:
                print("Erro", f"Erro ao listar alunos: {e}")
                tk.messagebox.showerror(
                    "Erro",
                    f"Possivelmente existe um erro na Coluna ID, verifique se todos os valores são do tipo inteiro, únicos e não vazios.",
                )

            if not self.students_tree.get_children():
                print("No students found within the specified probability range.")
        else:
            print("Dados não disponíveis ou não carregados.")

    def save_results(self):
        import os

        file_path = self.file_path_var.get()
        directory, filename = os.path.split(file_path)
        filename_without_ext = os.path.splitext(filename)[0]
        new_filename = f"{filename_without_ext}_ComProbabilidades.xlsx"
        save_path = os.path.join("data", new_filename)

        try:
            # Add dropout probabilities to the DataFrame
            self.data_model.add_dropout_probabilities_to_dataframe()

            # Save the DataFrame with the new probabilities column
            self.data_model.data.to_excel(save_path, index=False, engine="openpyxl")
            print(f"File saved: {save_path}")
            tk.messagebox.showinfo("Success", f"Arquivo salvo em: {save_path}")
        except Exception as e:
            print(f"Error saving file: {e}")
            tk.messagebox.showerror(
                "Error", f"Falha ao salvar (verifique se o arquivo não está aberto) {e}"
            )

    def setup_tab5(self):
        # Clear existing widgets in tab5
        for widget in self.tab5.winfo_children():
            widget.destroy()

        # Creating a bold font style for the title
        title_font = ("Helvetica", 10, "bold")

        # Creating a title label instead of a button
        title_label = ttk.Label(
            self.tab5,
            text="Lista de Alunos e sua Probabilidade de Evasão",
            font=title_font,
        )
        title_label.pack(pady=(10, 0))

        # Frame to hold the Treeview and its Scrollbar
        tree_frame = ttk.Frame(self.tab5)
        tree_frame.pack(pady=10, expand=True, fill="both")

        # Create a Treeview widget within the frame
        self.students_tree = ttk.Treeview(
            tree_frame, columns=("ID", "Probabilidade (%)"), show="headings"
        )
        self.students_tree.heading("ID", text="ID")
        self.students_tree.heading("Probabilidade (%)", text="Probabilidade (%)")

        self.students_tree.column("ID", anchor="center")
        self.students_tree.column("Probabilidade (%)", anchor="center")

        # Create a Scrollbar widget within the frame
        tree_scroll = ttk.Scrollbar(
            tree_frame, orient="vertical", command=self.students_tree.yview
        )
        tree_scroll.pack(side="right", fill="y")

        # Configure the Treeview to use the scrollbar
        self.students_tree.configure(yscrollcommand=tree_scroll.set)

        # Pack the Treeview last so it fills the remaining space except for the scrollbar
        self.students_tree.pack(side="left", expand=True, fill="both")

        # Configure the columns with the sorting function
        for col in self.students_tree["columns"]:
            self.students_tree.heading(
                col,
                text=col,
                command=lambda _col=col: self.treeview_sort_column(
                    self.students_tree, _col, True
                ),
            )

        # Probability threshold input fields
        # Frame for probability threshold inputs
        prob_threshold_frame = ttk.Frame(self.tab5)
        prob_threshold_frame.pack(pady=5, padx=10, fill="x")

        # Intermediate frame to hold inputs and labels, which can be centered more easily
        inputs_frame = ttk.Frame(prob_threshold_frame)
        inputs_frame.pack(anchor="center")

        # Min Probability Input
        ttk.Label(inputs_frame, text="Prob. Mínima (%):").pack(side="left", padx=(0, 5))
        self.min_prob_input = ttk.Entry(inputs_frame, width=6)
        self.min_prob_input.pack(side="left")
        self.min_prob_input.insert(0, "20.0")  # Default value

        # Spacer label for "até"
        ttk.Label(inputs_frame, text="até").pack(side="left", padx=5)

        # Max Probability Input
        ttk.Label(inputs_frame, text="Prob. Máxima (%):").pack(side="left", padx=(5, 0))
        self.max_prob_input = ttk.Entry(inputs_frame, width=6)
        self.max_prob_input.pack(side="left")
        self.max_prob_input.insert(0, "100.0")  # Default value

        # Variable to track the checkbox state
        self.show_non_dropout_var = tk.BooleanVar(value=True)
        # Checkbox for filtering non-dropout students
        self.show_non_dropout_checkbox = ttk.Checkbutton(
            self.tab5,
            text="Listar apenas alunos que não evadiram (Variável Dependente é 0).",
            variable=self.show_non_dropout_var,
            onvalue=True,
            offvalue=False,
        )
        self.show_non_dropout_checkbox.pack(pady=(0, 5))

        # Button to List Students, now positioned below the probability inputs
        list_students_button = ttk.Button(
            self.tab5,
            text="Listar Alunos",
            command=self.list_students_within_probability_range,
        )
        list_students_button.pack(pady=5)

    def treeview_sort_column(self, tv, col, reverse):
        l = [(tv.set(k, col), k) for k in tv.get_children("")]
        try:
            # Convert to float for numerical sorting
            l.sort(key=lambda t: float(t[0].replace("%", "")), reverse=reverse)
        except ValueError:
            # In case some data cannot be converted to float, fallback to string sort
            l.sort(reverse=reverse)

        for index, (val, k) in enumerate(l):
            tv.move(k, "", index)

        # Toggle the sorting order for the next click
        tv.heading(
            col,
            command=lambda _col=col: self.treeview_sort_column(tv, _col, not reverse),
        )


if __name__ == "__main__":
    root = tk.Tk()
    gui = DataScienceGUI(root)
    root.mainloop()

# TODO
# [X] Ao invés de usar 7 neuronios fixos na camada oculta, testar usando 2/3 do numero de entradas + o tamanho da saida. //Jeff Heaton, o autor de Introduction to Neural Networks for Java
# [X] Imprimir Quantidade de Neuronios e Camadas Ocultas.
# [X] Informar que arquivo foi carregado.
# [X] Informar que variáveis foram selecionadas.

# [X] Ao clicar em Buscar por ID é necessário re-calcular a % de evasao tambem.
# [X] Tentar Rodar a rede com varios valores de treinamento aleatorios repetidas vezes até atingir os menores erros de entropia (Limitar Qtd de Vezes).

# [X] Nova aba com a lista dos alunos que ainda não evadiram mas tem considerável probabilidade entre Z% e W%.
# [ ] Remover impressoes de debug.
# [ ] Fazer verificação e tratamento de erros.
# [ ] Refatorar codigo para organziação e legibilidade. (mudar nome main)

# [ ] Criar README.md e "Gitignore se necessário" e enviar para github.
# [ ] Melhorar a interface gráfica, sugestão de Vídeo sobre Custom Tkinter https://www.youtube.com/watch?v=rQLO1m8oia4


# [ ] Arrumar Sorting By ID, não está funcionando.
# [ ] Reativar botão para nao estar desabilitado após rodar uma vez (ver erro de % das variaveis que ficam 0.01% em algums casos) Arrumar tambem para nao gerar duplica/tripla lista de alunos.
# ## Button to Run Model
#     run_model_button = ttk.Button(
#         self.tab3, text="Executar Rede Neural", command=self.run_model
#     )
# [ ] Ser Possível de Selecionar Covariaveis com Shift click.
