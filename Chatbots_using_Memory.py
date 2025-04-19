# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 09:21:49 2024

@author: HP
"""
# Ex1 Memory and Chatbot
from langchain_openai import OpenAI
from langchain import ConversationChain

llm = OpenAI(temperature = 0)
conversation = ConversationChain(llm = llm, verbose = True)

conversation.predict(input="Hi !")
conversation.predict(input="Can we talk about the weather?")
print(conversation.predict(input = "Its a beautiful day"))

#Ex2 Terminal Chatbot

from langchain_openai import OpenAI
from langchain import ConversationChain

llm = OpenAI(temperature = 0)
conversation = ConversationChain(llm = llm, verbose = True)
#Make verbose = False, if you don't want to have a clean UX

print("Welcome to your AI Chatbot!!")
for i in range(3):
    human_input = input("You:")
    ai_response = conversation.predict(input = human_input)
    print(f"AI: {ai_response}")


#Ex3 Store and retrieve the chat history - context information using chat message history class

from langchain_openai import OpenAI
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict
#import langchain_core.messages.base 

#Initialize an empty history object which has two methods "add user message" and "add ai message"
initial_dialogue = ChatMessageHistory()

#create a chat history and save it in a dictionary. This will be the context of the following chat
initial_dialogue.add_user_message("Hi! Lets talk about giraffes")
initial_dialogue.add_ai_message("I am down to talk!!")
print(initial_dialogue.messages)
dicts = messages_to_dict(initial_dialogue.messages)
#print(dicts)

#create an object to this conversation history which is the context of the following chat
context = messages_from_dict(dicts)

#INitialize the Chat history object with the context information
llm = OpenAI(temperature = 0)
history = ChatMessageHistory(messages = context)

#INitialize a conversation buffer memory object and load the history object
buffer = ConversationBufferMemory(chat_memory = history)

#Initialize a conversation chain

conversation = ConversationChain(llm = llm, memory = buffer, verbose = True)

#begin the conversation. The AI will respond remembering the context stored in the buffer
print(conversation.predict(input = "What are they?"))

















