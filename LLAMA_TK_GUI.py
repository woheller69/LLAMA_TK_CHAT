#!/usr/bin/env python3

import tkinter as tk
from tkinter import scrolledtext
import threading
import time
import ctypes
from typing_extensions import Annotated
import typer

from llama_cpp import Llama
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType

class thread_with_exception(threading.Thread):
    def __init__(self, name, chat_gui_instance):
        threading.Thread.__init__(self)
        self.name = name
        self.chat_gui_instance = chat_gui_instance
             
    def run(self):
        try:
            self.chat_gui_instance.init_inference()
        finally:
            print('Inference terminated')
          
    def get_id(self):
 
        # returns id of the respective thread
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id
  
    def raise_exception(self):
        thread_id = self.get_id()
        if thread_id != None:
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), ctypes.py_object(SystemExit))
            if res > 1:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), 0)
                print('Exception raise failure')

class ChatGUI:
    def __init__(self):
        self.main_model = None
        self.model_path = None
        self.llama_cpp_agent = None
        self.sysprompt = None
        self.threads = 0
        self.context = 0
        self.chatformat = None
        self.output_window = None
        self.input_text = None
        self.token_count = 0
        self.start_time = 0
        self.prompt_eval_time = 0
        self.inference_thread = None
        self.model_reply = ""
        self.root = None
        self.debug = True

    def run(self):
        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.title('LLAMA TK GUI')
        self.root.geometry("1024x768")

        self.input_text = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, height=5)
        self.input_text.pack(side='bottom', fill='both', expand=True)

        self.output_window = scrolledtext.ScrolledText(self.root, wrap=tk.WORD)
        self.output_window.pack(side='top', fill='both', expand=True)

        generate_button = tk.Button(self.root, text="Generate", command=self.generate)
        stop_button = tk.Button(self.root, text="Stop", command=self.stop)
        exit_button = tk.Button(self.root, text="Exit", command=self.root.destroy)
        newchat_button = tk.Button(self.root, text="New Chat", command=self.new_chat)

        generate_button.pack(side='left', padx=(20, 0))
        stop_button.pack(side='left', padx=(20, 0))
        exit_button.pack(side='right', padx=(0, 20))
        newchat_button.pack(side='right', padx=(0, 20))
        
        self.output_window.insert(tk.END, "\nModel: " + self.model_path)
        self.output_window.insert(tk.END, "\nUsing " + repr(self.threads) + " threads")

        self.output_window.insert(tk.END, "\nSystem prompt: " + self.sysprompt)
        self.output_window.insert(tk.END, "\nContext length: " + repr(self.context))
        self.output_window.insert(tk.END, "\n\n")
        self.root.after(10, lambda: self.input_text.focus_set())
        self.main_model = Llama(
            model_path=self.model_path,
            n_gpu_layers=0,
            f16_kv=True,
            use_mmap=True,
            use_mlock=False,
            embedding=False,
            n_threads=self.threads,
            n_batch=128,
            n_ctx=self.context,
            offload_kqv=False,
            last_n_tokens_size=1024,
            verbose=True,
            seed=-1,
        )
        self.llama_cpp_agent = LlamaCppAgent(self.main_model, debug_output=self.debug,
                                            system_prompt=self.sysprompt,
                                            predefined_messages_formatter_type=self.resolve_formatter(self.chatformat))
        self.root.mainloop()


    def opt(self, model: Annotated[str, typer.Option("--model", "-m", help="Model to use for chatbot")] = None,
            n_threads: Annotated[int, typer.Option("--n-threads", "-t", help="Number of threads to use for chatbot")] = 4,
            template: Annotated[str, typer.Option("--format", "-f", help="Prompt template format for the chatbot, e.g. CHATML, ALPACA,... ")] = "CHATML",
            sys: Annotated[str, typer.Option("--sysprompt", "-s", help="System prompt to use for chatbot")] = "",
            ctx: Annotated[int, typer.Option("--context-length", "-c", help="Context length")] = 2048):

        self.sysprompt = sys
        self.chatformat = template
        self.context = ctx
        self.threads = n_threads

        if model is None:
            print("Specify model with --model or -m")
            quit()
        
        self.model_path = model

        self.run()

    def resolve_formatter(self, enum_str):
        for enum_member in MessagesFormatterType:
            if enum_member.name == enum_str:
                return enum_member
        raise ValueError("Enum member not found")

    def streaming_callback(self, response):
        self.token_count += 1
        if self.token_count == 1:
            self.prompt_eval_time = time.time() - self.start_time
            self.start_time = time.time()
            self.output_window.insert(tk.END, "AI:")

        self.output_window.insert(tk.END, response.text)
        self.output_window.yview(tk.END)
        self.model_reply += response.text

        if response.is_last_response == True:
            self.print_token_speed()

        self.root.update_idletasks()

    def print_token_speed(self):
        if self.token_count > 0:
            end_time = time.time()
            tokens_per_second = (self.token_count - 1) / (end_time - self.start_time)
            self.output_window.insert(tk.END, f"\n\nPrompt evaluation: {self.prompt_eval_time:.2f} seconds")
            self.output_window.insert(tk.END, f"\nTokens: {self.token_count}  Tokens/second: {tokens_per_second:.2f}")
        self.output_window.insert(tk.END, "\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n\n")
        self.output_window.yview(tk.END)

    def inference(self, user_input):
        self.start_time = time.time()

        message = user_input
        self.output_window.insert(tk.END, "\n<<<<<<<<<<<<<<< AI <<<<<<<<<<<<<<<\n\n")
        self.output_window.yview(tk.END)
        self.token_count = 0
        self.llama_cpp_agent.get_chat_response(
            user_input,
            temperature=0.7,
            top_k=40,
            top_p=0.4,
            repeat_penalty=1.18,
            repeat_last_n=64,
            max_tokens=2000,
            stream=True,
            print_output=False,
            streaming_callback=self.streaming_callback
        )

    def init_inference(self):
        self.output_window.insert(tk.END, "\n>>>>>>>>>>>>>> USER >>>>>>>>>>>>>>\n\n")
        self.output_window.insert(tk.END, self.input_text.get("1.0", "end-1c"))
        self.output_window.insert(tk.END, "\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n")
        self.output_window.yview(tk.END)
        message = self.input_text.get("1.0", "end-1c")
        self.input_text.delete("1.0", "end")
        self.inference(message)

    def generate(self):
        self.model_reply = ""
        self.inference_thread = thread_with_exception("InferenceThread", self)
        self.inference_thread.start()

    def stop(self):
        if self.inference_thread is not None:
            if self.inference_thread.get_id() is not None:
                self.inference_thread.raise_exception()
                self.inference_thread.join()
                self.print_token_speed()
                print(self.model_reply)
                self.llama_cpp_agent.add_message(self.model_reply, "assistant")
                self.llama_cpp_agent.save_messages("msg.txt")
                  
    def new_chat(self):
        del self.llama_cpp_agent
        self.output_window.delete('1.0', tk.END)
        self.llama_cpp_agent = LlamaCppAgent(self.main_model, debug_output=self.debug,
                                            system_prompt=self.sysprompt,
                                            predefined_messages_formatter_type=self.resolve_formatter(self.chatformat))

    def on_closing(self):
        self.root.destroy()

if __name__ == "__main__":
    gui = ChatGUI()
    typer.run(gui.opt)
