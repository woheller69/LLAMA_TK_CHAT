#!/usr/bin/env python3

from typing_extensions import Annotated
import typer
import tkinter as tk
from tkinter import scrolledtext
import threading
import time
import ctypes

from llama_cpp import Llama
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType

main_model = None
llama_cpp_agent = None
prompt = None
sysprompt = None
chatformat = None
output_window = None
input_text = None
token_count = 0
start_time = 0
prompt_eval_time = 0
inference_thread = None


# create typer app
app = typer.Typer()


class thread_with_exception(threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name
             
    def run(self):
 
        # target function of the thread class
        try:
            init_inference()
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

def opt(
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Model to use for chatbot"),
    ] = None,
    n_threads: Annotated[
        int,
        typer.Option("--n-threads", "-t", help="Number of threads to use for chatbot"),
    ] = 4,
    template: Annotated[
        str,
        typer.Option("--format", "-f", help="Prompt template format for the chatbot, e.g. CHATML, ALPACA,... "),
    ] = "CHATML",
    sys: Annotated[
        str,
        typer.Option("--sysprompt", "-s", help="System prompt to use for chatbot"),
    ] = "",
    ctx: Annotated[
        int,
        typer.Option("--context-length", "-c", help="Context length"),
    ] = 2048,
):
    global main_model
    global llama_cpp_agent
    global output_window
    global sysprompt
    global chatformat 
    
    sysprompt = sys
    chatformat = template
    
    if model is None:
        print("Specify model with --model or -m")
        quit()

    #For instances of Llama class of llama-cpp-python
    main_model = Llama(
        model_path=model,
        n_gpu_layers=0,
        f16_kv=True,
        use_mmap=True,
        use_mlock=False,
        embedding=False,
        n_threads=n_threads,
        n_batch=128,
        n_ctx=ctx,
        offload_kqv=False,
        last_n_tokens_size=1024,
        verbose=True,
        seed=-1,
    )
    llama_cpp_agent = LlamaCppAgent(main_model, debug_output=False,
                                system_prompt=sysprompt,
                                predefined_messages_formatter_type=resolve_formatter(chatformat))
                               

    output_window.insert(tk.END, "\nModel: "+model)
    output_window.insert(tk.END, "\nUsing " + repr(n_threads) + " threads")

    output_window.insert(tk.END, "\nSystem prompt: " + sysprompt)
    output_window.insert(tk.END, "\nContext length: " + repr(ctx))
    output_window.insert(tk.END, "\n\n")
    root.after(10, lambda: input_text.focus_set())
    root.mainloop()

def resolve_formatter(enum_str):
    for enum_member in MessagesFormatterType:
        if enum_member.name == enum_str:
            return enum_member
    raise ValueError("Enum member not found")

def streaming_callback(response):
    global token_count
    global output_window
    global start_time
    global prompt_eval_time
   
    token_count += 1
    if token_count == 1:
        prompt_eval_time = time.time() - start_time 
        start_time = time.time() 
        output_window.insert(tk.END, "AI:")
        
    output_window.insert(tk.END, response.text)
    output_window.yview(tk.END)

    if response.is_last_response == True:
        print_token_speed()
   
    root.update_idletasks()

def print_token_speed():
    global token_count
    global output_window
    global start_time
    global prompt_eval_time
    if token_count > 0:
        end_time = time.time()
        tokens_per_second = (token_count -1) / (end_time - start_time)
        output_window.insert(tk.END, f"\n\nPrompt evaluation: {prompt_eval_time:.2f} seconds")     
        output_window.insert(tk.END, f"\nTokens: {token_count}  Tokens/second: {tokens_per_second:.2f}")                
    output_window.insert(tk.END, "\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n\n")
    output_window.yview(tk.END)

def inference(user_input):
    global output_window
    global token_count
    global start_time

    start_time = time.time()
        
    message = user_input
    # execute chat completion and ignore the full response since 
    # we are outputting it incrementally
    output_window.insert(tk.END, "\n<<<<<<<<<<<<<<< AI <<<<<<<<<<<<<<<\n\n")
    output_window.yview(tk.END) 
    token_count=0
    llama_cpp_agent.get_chat_response(
        user_input, 
        temperature=0.7, 
        top_k=40, 
        top_p=0.4,
        repeat_penalty=1.18, 
        repeat_last_n=64, 
        max_tokens=2000,
        stream=True,
        print_output=False,
        streaming_callback=streaming_callback
    )
      

def init_inference():
    global input_text
    global output_window
    # Copy and paste into output window
    output_window.insert(tk.END, "\n>>>>>>>>>>>>>> USER >>>>>>>>>>>>>>\n\n")
    output_window.insert(tk.END, input_text.get("1.0", "end-1c"))
    output_window.insert(tk.END, "\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n")
    output_window.yview(tk.END) 
    message = input_text.get("1.0", "end-1c")
    input_text.delete("1.0", "end") 
    inference(message)


def generate():
    global inference_thread
    inference_thread = thread_with_exception('Inference')
    inference_thread.start()

def stop():
    global inference_thread
    if inference_thread != None:
        print_token_speed()
        inference_thread.raise_exception()
        inference_thread.join()
    
def exit():
    quit()
    
def newchat():
    global llama_cpp_agent
    global main_model
    global sysprompt
    global chatformat
    global output_window
    del llama_cpp_agent
    output_window.delete('1.0', tk.END)
    llama_cpp_agent = LlamaCppAgent(main_model, debug_output=False,
                                system_prompt=sysprompt,
                                predefined_messages_formatter_type=resolve_formatter(chatformat))


def on_closing():
    root.destroy()

if __name__ == "__main__":
    # Setup Tkinter GUI
    root = tk.Tk()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    root.title('LLAMA TK GUI')
    root.geometry("1024x768")

    input_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=5)
    input_text.pack(side='bottom', fill='both', expand=True)

    output_window = scrolledtext.ScrolledText(root, wrap=tk.WORD)
    output_window.pack(side='top', fill='both', expand=True)

    generate_button = tk.Button(root, text="Generate", command=generate)
    stop_button = tk.Button(root, text="Stop", command=stop)
    exit_button = tk.Button(root, text="Exit", command=exit)
    newchat_button = tk.Button(root, text="New Chat", command=newchat)

    generate_button.pack(side='left', padx=(20, 0))
    stop_button.pack(side='left', padx=(20, 0))
    exit_button.pack(side='right', padx=(0, 20))
    newchat_button.pack(side='right', padx=(0, 20))

    typer.run(opt)
