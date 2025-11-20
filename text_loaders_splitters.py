#!/usr/bin/env python
# coding: utf-8

# ## Document Loaders In LangChain

# #### TextLoader

# In[1]:


from langchain.document_loaders import TextLoader

loader = TextLoader("nvda_news_1.txt")
loader.load()


# In[2]:


type(loader)


# In[3]:


loader.file_path


# #### CSVLoader

# In[4]:


from langchain.document_loaders.csv_loader import CSVLoader


# In[5]:


loader = CSVLoader(file_path="movies.csv")
data = loader.load()
data


# In[6]:


data[0]


# In[7]:


loader = CSVLoader(file_path="movies.csv", source_column="title")
data = loader.load()
data


# In[8]:


data[0].page_content


# In[9]:


data[0].metadata


# #### UnstructuredURLLoader

# UnstructuredURLLoader of Langchain internally uses unstructured python library to load the content from url's
# 
# https://unstructured-io.github.io/unstructured/introduction.html
# 
# https://pypi.org/project/unstructured/#description

# In[11]:


#installing necessary libraries, libmagic is used for file type detection
get_ipython().system('pip3 install unstructured libmagic python-magic python-magic-bin')


# In[10]:


from langchain.document_loaders import UnstructuredURLLoader


# In[11]:


loader = UnstructuredURLLoader(
    urls = [
        "https://www.moneycontrol.com/news/business/banks/hdfc-bank-re-appoints-sanmoy-chakrabarti-as-chief-risk-officer-11259771.html",
        "https://www.moneycontrol.com/news/business/markets/market-corrects-post-rbi-ups-inflation-forecast-icrr-bet-on-these-top-10-rate-sensitive-stocks-ideas-11142611.html"
    ]
)


# In[12]:


data = loader.load()
len(data)


# In[13]:


data[0].page_content[0:100]


# In[14]:


data[0].metadata


# ## Text Splitters

# Why do we need text splitters in first place?
# 
# LLM's have token limits. Hence we need to split the text which can be large into small chunks so that each chunk size is under the token limit. There are various text splitter classes in langchain that allows us to do this.

# In[80]:


# Taking some random text from wikipedia

text = """Interstellar is a 2014 epic science fiction film co-written, directed, and produced by Christopher Nolan. 
It stars Matthew McConaughey, Anne Hathaway, Jessica Chastain, Bill Irwin, Ellen Burstyn, Matt Damon, and Michael Caine. 
Set in a dystopian future where humanity is embroiled in a catastrophic blight and famine, the film follows a group of astronauts who travel through a wormhole near Saturn in search of a new home for humankind.

Brothers Christopher and Jonathan Nolan wrote the screenplay, which had its origins in a script Jonathan developed in 2007 and was originally set to be directed by Steven Spielberg. 
Kip Thorne, a Caltech theoretical physicist and 2017 Nobel laureate in Physics,[4] was an executive producer, acted as a scientific consultant, and wrote a tie-in book, The Science of Interstellar. 
Cinematographer Hoyte van Hoytema shot it on 35 mm movie film in the Panavision anamorphic format and IMAX 70 mm. Principal photography began in late 2013 and took place in Alberta, Iceland, and Los Angeles. 
Interstellar uses extensive practical and miniature effects, and the company Double Negative created additional digital effects.

Interstellar premiered in Los Angeles on October 26, 2014. In the United States, it was first released on film stock, expanding to venues using digital projectors. The film received generally positive reviews from critics and grossed over $677 million worldwide ($715 million after subsequent re-releases), making it the tenth-highest-grossing film of 2014. 
It has been praised by astronomers for its scientific accuracy and portrayal of theoretical astrophysics.[5][6][7] Interstellar was nominated for five awards at the 87th Academy Awards, winning Best Visual Effects, and received numerous other accolades."""


# #### Manual approach of splitting the text into chunks

# In[81]:


# Say LLM token limit is 100, in that case we can do simple thing such as this

text[0:100]


# In[82]:


# Well but we want complete words and want to do this for entire text, may be we can use Python's split funciton

words = text.split(" ")
len(words)


# In[83]:


chunks = []

s = ""
for word in words:
    s += word + " "
    if len(s)>200:
        chunks.append(s)
        s = ""
        
chunks.append(s)


# In[84]:


chunks[:2]


# **Splitting data into chunks can be done in native python but it is a tidious process. Also if necessary, you may need to experiment with various delimiters in an iterative manner to ensure that each chunk does not exceed the token length limit of the respective LLM.**
# 
# **Langchain provides a better way through text splitter classes.**

# #### Using Text Splitter Classes from Langchain
# 
# #### CharacterTextSplitter

# In[85]:


from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size=200,
    chunk_overlap=0
)


# In[86]:


chunks = splitter.split_text(text)
len(chunks)


# In[87]:


for chunk in chunks:
    print(len(chunk))


# As you can see, all though we gave 200 as a chunk size since the split was based on \n, it ended up creating chunks that are bigger than size 200. 

# Another class from Langchain can be used to recursively split the text based on a list of separators. This class is RecursiveTextSplitter. Let's see how it works

# #### RecursiveTextSplitter

# In[88]:


text


# In[117]:


from langchain.text_splitter import RecursiveCharacterTextSplitter

r_splitter = RecursiveCharacterTextSplitter(
    separators = ["\n\n", "\n", " "],  # List of separators based on requirement (defaults to ["\n\n", "\n", " "])
    chunk_size = 200,  # size of each chunk created
    chunk_overlap  = 0,  # size of  overlap between chunks in order to maintain the context
    length_function = len  # Function to calculate size, currently we are using "len" which denotes length of string however you can pass any token counter)
)


# In[118]:


chunks = r_splitter.split_text(text)

for chunk in chunks:
    print(len(chunk))


# **Let's understand how exactly it formed these chunks**

# In[109]:


first_split = text.split("\n\n")[0]
first_split


# In[110]:


len(first_split)


# Recursive text splitter uses a list of separators, i.e.  separators = ["\n\n", "\n", "."]
# 
# So now it will first split using \n\n and then if the resulting chunk size is greater than the chunk_size parameter which is 200
# in our case, then it will use the next separator which is \n

# In[119]:


second_split = first_split.split("\n")
second_split


# In[120]:


for split in second_split:
    print(len(split))


# Third split exceeds chunk size 200. Now it will further try to split that using the third separator which is ' ' (space)

# In[115]:


second_split[2]


# When you split this using space (i.e. second_split[2].split(" ")), it will separate out each word and then it will merge those 
# chunks such that their size is close to 200

# <img src="chunk_size.jpg"/>
