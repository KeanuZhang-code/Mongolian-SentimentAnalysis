import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json, re, os, time, random, jieba, pickle, shutil, datetime
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# 确保中文显示正常
import matplotlib

matplotlib.use('Agg')  # 非交互式后端，避免Tkinter冲突


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.minsize(900, 600)  # 设置最小窗口尺寸
        self.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 设置样式
        self.style = ttk.Style()
        self.style.configure("TButton", font=("SimHei", 10))
        self.style.configure("TLabel", font=("SimHei", 10))
        self.style.configure("Header.TLabel", font=("SimHei", 12, "bold"))

        self.create_widgets()
        self.create_layout()

        # 状态变量
        self.current_file = None
        self.model_loaded = False
        self.load_model()  # 尝试加载模型

    def create_widgets(self):
        # 创建主菜单栏
        menubar = tk.Menu(self.master)

        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="新建", accelerator="Ctrl+N",
                              command=self.new_file, compound=tk.LEFT)
        file_menu.add_command(label="打开", accelerator="Ctrl+O",
                              command=self.open_file, compound=tk.LEFT)
        file_menu.add_command(label="保存", accelerator="Ctrl+S",
                              command=self.save_file, compound=tk.LEFT)
        file_menu.add_separator()
        file_menu.add_command(label="退出", accelerator="Ctrl+Q",
                              command=self.master.quit, compound=tk.LEFT)

        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="关于", command=self.show_about)
        help_menu.add_command(label="使用帮助", command=self.show_help)

        # 添加菜单到菜单栏
        menubar.add_cascade(label="文件", menu=file_menu)
        menubar.add_cascade(label="帮助", menu=help_menu)

        # 设置菜单栏
        self.master.config(menu=menubar)

        # 绑定快捷键
        self.master.bind("<Control-n>", lambda e: self.new_file())
        self.master.bind("<Control-o>", lambda e: self.open_file())
        self.master.bind("<Control-s>", lambda e: self.save_file())
        self.master.bind("<Control-q>", lambda e: self.master.quit())

    def create_layout(self):
        # 创建主容器，使用网格布局
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧输入区域
        left_frame = ttk.LabelFrame(main_frame, text="蒙古语文本输入", padding=10)
        left_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # 右侧输出区域
        right_frame = ttk.LabelFrame(main_frame, text="情感分析结果", padding=10)
        right_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        # 底部按钮区域
        bottom_frame = ttk.Frame(main_frame, padding=10)
        bottom_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # 配置网格权重，使区域可以拉伸
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # 输入文本框及滚动条
        input_frame = ttk.Frame(left_frame)
        input_frame.pack(fill=tk.BOTH, expand=True)

        self.text_input = tk.Text(input_frame, wrap=tk.WORD, font=("SimHei", 10),
                                  borderwidth=1, relief=tk.SUNKEN)
        self.text_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        input_scroll = ttk.Scrollbar(input_frame, command=self.text_input.yview)
        input_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_input.config(yscrollcommand=input_scroll.set)

        # 输出文本框及滚动条
        output_frame = ttk.Frame(right_frame)
        output_frame.pack(fill=tk.BOTH, expand=True)

        self.text_output = tk.Text(output_frame, wrap=tk.WORD, font=("SimHei", 10),
                                   borderwidth=1, relief=tk.SUNKEN, state=tk.DISABLED)
        self.text_output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        output_scroll = ttk.Scrollbar(output_frame, command=self.text_output.yview)
        output_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_output.config(yscrollcommand=output_scroll.set)

        # 情感分析按钮
        self.analyze_btn = ttk.Button(bottom_frame, text="开始情感分析",
                                      command=self.analyze_sentiment, padding=5)
        self.analyze_btn.pack(side=tk.LEFT, padx=10)

        # 清空按钮
        self.clear_btn = ttk.Button(bottom_frame, text="清空内容",
                                    command=self.clear_content, padding=5)
        self.clear_btn.pack(side=tk.LEFT, padx=10)

        # 状态标签
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_label = ttk.Label(bottom_frame, textvariable=self.status_var, anchor=tk.E)
        status_label.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=10)

    def load_model(self):
        """尝试加载模型和相关信息"""
        try:
            self.status_var.set("正在加载模型...")
            self.master.update()  # 更新界面显示

            # 加载信息
            self.num2label, self.tokenizer, self.maxlen, self.word2num = pickle.load(
                open('model/final_info.pkl', 'rb'))
            # 加载模型
            self.model = load_model('model/model.h5')

            self.model_loaded = True
            self.status_var.set("模型加载成功，就绪")
        except Exception as e:
            self.model_loaded = False
            self.status_var.set(f"模型加载失败: {str(e)}")
            messagebox.showerror("错误", f"模型加载失败: {str(e)}\n请确保model目录存在且包含必要文件")

    def predict_text_label(self, text):
        """预测文本情感"""
        if not self.model_loaded:
            raise Exception("模型未加载成功，无法进行预测")

        # 分词
        texts = [text]
        text_vector = self.tokenizer.texts_to_sequences(texts)
        sequences_pad = pad_sequences(
            text_vector,
            maxlen=self.maxlen,
            padding='post',
            truncating='post',
            value=0.0,
            dtype='int32',
        )
        label = self.model.predict(sequences_pad).argmax(axis=1)[0]
        return self.num2label[label]

    def analyze_sentiment(self):
        """情感分析按钮点击事件"""
        input_text = self.text_input.get("1.0", tk.END).strip()

        if not input_text:
            messagebox.showwarning("警告", "请输入要分析的蒙古语文本")
            return

        if not self.model_loaded:
            messagebox.showerror("错误", "模型未加载成功，无法进行分析")
            return

        try:
            self.status_var.set("正在分析情感...")
            self.analyze_btn.config(state=tk.DISABLED)
            self.master.update()  # 更新界面

            # 执行情感分析
            result = self.predict_text_label(input_text)

            # 显示结果
            self.text_output.config(state=tk.NORMAL)
            self.text_output.delete("1.0", tk.END)

            # 根据情感结果添加颜色标识
            self.text_output.insert(tk.END, f"情感类别: {result}\n\n")

            # 添加情感强度说明（示例）
            sentiment_info = {
                "高兴": "表示积极、愉悦的情感",
                "生气": "表示愤怒、不满的情感",
                "欣赏": "表示赞赏、认同的情感",
                "厌烦": "表示厌倦、反感的情感",
                "害怕": "表示恐惧、担忧的情感",
                "忧愁": "表示忧虑、愁苦的情感",
                "惊吓": "表示受惊、惊吓的情感"
            }

            if result in sentiment_info:
                self.text_output.insert(tk.END, f"情感说明: {sentiment_info[result]}")
            else:
                self.text_output.insert(tk.END, "情感说明: 该情感类别未定义")

            self.text_output.config(state=tk.DISABLED)
            self.status_var.set("情感分析完成")

        except Exception as e:
            self.status_var.set(f"分析失败: {str(e)}")
            messagebox.showerror("错误", f"分析过程中出错: {str(e)}")
        finally:
            self.analyze_btn.config(state=tk.NORMAL)

    def clear_content(self):
        """清空输入输出内容"""
        self.text_input.delete("1.0", tk.END)
        self.text_output.config(state=tk.NORMAL)
        self.text_output.delete("1.0", tk.END)
        self.text_output.config(state=tk.DISABLED)
        self.status_var.set("内容已清空")

    # 文件操作函数
    def new_file(self):
        """新建文件"""
        if self.text_input.edit_modified():
            response = messagebox.askyesnocancel("保存", "是否保存当前内容?")
            if response is None:  # 取消
                return
            if response:  # 是
                self.save_file()

        self.text_input.delete("1.0", tk.END)
        self.text_output.config(state=tk.NORMAL)
        self.text_output.delete("1.0", tk.END)
        self.text_output.config(state=tk.DISABLED)
        self.current_file = None
        self.master.title("蒙古语情感分析工具")
        self.status_var.set("新建文件")

    def open_file(self):
        """打开文件"""
        file_path = filedialog.askopenfilename(
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")],
            title="打开文件"
        )

        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()

                self.text_input.delete("1.0", tk.END)
                self.text_input.insert(tk.END, content)
                self.current_file = file_path
                self.master.title(f"蒙古语情感分析工具 - {os.path.basename(file_path)}")
                self.status_var.set(f"已打开文件: {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"打开文件失败: {str(e)}")
                self.status_var.set("打开文件失败")

    def save_file(self):
        """保存文件"""
        if self.current_file:
            try:
                content = self.text_input.get("1.0", tk.END)
                with open(self.current_file, 'w', encoding='utf-8') as file:
                    file.write(content)
                self.text_input.edit_modified(False)
                self.status_var.set(f"已保存文件: {self.current_file}")
                return True
            except Exception as e:
                messagebox.showerror("错误", f"保存文件失败: {str(e)}")
                self.status_var.set("保存文件失败")
                return False
        else:
            return self.save_file_as()

    def save_file_as(self):
        """另存为文件"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")],
            title="另存为"
        )

        if file_path:
            self.current_file = file_path
            self.master.title(f"蒙古语情感分析工具 - {os.path.basename(file_path)}")
            return self.save_file()
        return False

    # 帮助菜单函数
    def show_about(self):
        """显示关于对话框"""
        about_window = tk.Toplevel(self.master)
        about_window.title("关于")
        about_window.geometry("300x200")
        about_window.resizable(False, False)
        about_window.transient(self.master)  # 设置为主窗口的子窗口
        about_window.grab_set()  # 模态窗口

        ttk.Label(about_window, text="蒙古语情感分析工具", style="Header.TLabel").pack(pady=10)
        ttk.Label(about_window, text="版本: 1.0.0").pack(pady=5)
        ttk.Label(about_window, text="用于分析蒙古语文本的情感倾向").pack(pady=5)
        ttk.Label(about_window, text="支持情感类别: 高兴、生气、欣赏等").pack(pady=5)

        ttk.Button(about_window, text="确定", command=about_window.destroy).pack(pady=10)

    def show_help(self):
        """显示使用帮助"""
        help_window = tk.Toplevel(self.master)
        help_window.title("使用帮助")
        help_window.geometry("500x400")
        help_window.transient(self.master)
        help_window.grab_set()

        help_text = tk.Text(help_window, wrap=tk.WORD, font=("SimHei", 10),
                            borderwidth=1, relief=tk.SUNKEN, state=tk.DISABLED)
        help_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        scroll = ttk.Scrollbar(help_text, command=help_text.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        help_text.config(yscrollcommand=scroll.set)

        # 帮助内容
        content = """使用说明:
1. 在左侧文本框中输入或粘贴蒙古语文本
2. 点击"开始情感分析"按钮进行分析
3. 分析结果将显示在右侧文本框中

快捷键:
- Ctrl+N: 新建文件
- Ctrl+O: 打开文件
- Ctrl+S: 保存文件
- Ctrl+Q: 退出程序

功能说明:
- 可以打开和保存文本文件
- 分析结果包含情感类别和简要说明
- 支持多种情感类别的识别"""

        help_text.config(state=tk.NORMAL)
        help_text.insert(tk.END, content)
        help_text.config(state=tk.DISABLED)

        ttk.Button(help_window, text="关闭", command=help_window.destroy).pack(pady=10)


if __name__ == '__main__':
    root = tk.Tk()
    root.title("蒙古语情感分析工具")
    app = Application(master=root)
    root.mainloop()
