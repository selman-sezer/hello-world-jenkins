## 1 - Prompt matching with questions
Instead of term frequency–inverse document frequency (tf-idf) we use word2vec for feature extraction. 
#### 1.1
```
def preprocess_text(text):
    # Convert text to lowercase and split it into words
    words = text.lower().split()
    # Remove stop words
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return words
keywords = []
prompts = []
...
```
Breakdown of this code cell:
###### 1.1.1. Data Preprocessing:
The preprocess_text function is defined to convert text to lowercase, split it into words, and remove stop words.

###### 1.1.2. Data Preparation:
User prompts are extracted from a dataset (code2convos). It only extracts the prompts from the "user" role.

###### 1.1.3. Word2Vec Training:
Sentences (comprising both user prompts and questions) are preprocessed, and a Word2Vec model is trained using the Word2Vec class from the Gensim library. The model is configured with a vector size of 100, a window size of 5, and a minimum word count of 1.

###### 1.1.4. Feature Extraction for Questions:
For each question, the code obtains its Word2Vec vector by averaging the vectors of its preprocessed words. The resulting vectors are used to create a DataFrame (questions_word2vec).

###### 1.1.5. Feature Extraction for User Prompts:
For each code, the user prompts associated with that code are processed similarly to questions. Word2Vec vectors are obtained for each prompt, and DataFrames are created for each code (code2prompts_word2vec). Empty DataFrames are printed if there are no valid prompts for a code.

###### 1.1.6. Handling NaN Values:
NaN values in the DataFrames are replaced with zeros.


#### 1.2

Then, we calculate the cosine similarity between the Word2Vec representations of questions (questions_word2vec) and the average representations of user prompts associated with different codes (code2prompts_word2vec). The resulting similarity scores are then organized into a DataFrame (similarity_df) for further analysis.

```
from sklearn.metrics.pairwise import cosine_similarity

# Calculate cosine similarity between questions_word2vec and each code's prompts_word2vec
code2similarity = {}

for code, prompts_df in code2prompts_word2vec.items():
    similarity_scores = cosine_similarity(prompts_df, questions_word2vec)
    # Average similarity scores across prompts for each code
    avg_similarity_score = similarity_scores.mean(axis=0)
    code2similarity[code] = avg_similarity_score

# Create a DataFrame to store the similarity scores
similarity_df = pd.DataFrame(code2similarity, index=questions)

# Display the resulting DataFrame
print(similarity_df)
```
Here is the workflow of this part:
###### 1.2.1. Cosine Similarity Calculation:
The cosine_similarity function from scikit-learn is used to compute the cosine similarity between each code's prompts and all questions. This results in a matrix of similarity scores, where each row corresponds to a prompt for a specific code, and each column corresponds to a question.

###### 1.2.2. Averaging Similarity Scores:
For each code (user code), we calculate the average similarity score across all prompts associated with that code. This is done by taking the mean along the rows of the similarity matrix.

###### 1.2.3. Storing Results:
The average similarity scores for each code are stored in the code2similarity dictionary, where the code serves as the key.

###### 1.2.4. Creating a DataFrame:
The similarity scores are organized into a DataFrame (similarity_df). Each row of the DataFrame corresponds to a question, and each column corresponds to a code. The values in the DataFrame represent the average cosine similarity score between the questions and the prompts for each code.

###### 1.2.5. Displaying Results:
The resulting DataFrame is printed, showing the average similarity scores between questions and prompts for each code


#### 1.2

Next, we process the similarity DataFrame (similarity_df) obtained from the previous code section. It organizes and structures the similarity scores to create a new DataFrame (question_mapping_scores) that provides a mapping between codes and their respective similarity scores for each question. 

```
code2questionmapping = dict()
for code, cosine_scores in similarity_df.items():
    code2questionmapping[code] = similarity_df[code].tolist()

question_mapping_scores = pd.DataFrame(code2questionmapping).T
question_mapping_scores.reset_index(inplace=True)
question_mapping_scores.rename(columns={i: f"Q_{i}" for i in range(len(questions))}, inplace=True)
question_mapping_scores.rename(columns={"index" : "code"}, inplace=True)
question_mapping_scores
```
------------
------------
------------

## 2 - Feature Engineering 
To improve, we add following parts.

#### 2.1. 
We initialize a list of keywords (keywords2search)

#### 2.2. 
Then, this code cell processes conversations stored in code2convos and extracts various features related to user prompts and ChatGPT responses for each code. Additionally, it incorporates a pattern-based approach to identify if a user prompt contains specific error-related terms.

```
# Using pattern based approach in the structure of the sentences to tell if it is an error or not

code2features = defaultdict(lambda : defaultdict(int))
for code, convs in code2convos.items():
    if len(convs) == 0:
        print(code)
        continue
    for c in convs:
        text = c["text"].lower()
        if c["role"] == "user":
            # User Prompts
            # count the user prompts
            code2features[code]["#user_prompts"] += 1
            for kw in keywords2search:
                code2features[code][f"#{kw}"] +=  len(re.findall(rf"\b{kw}\b", text))
            code2features[code]["prompt_avg_chars"] += len(text)
        else:
            # ChatGPT Responses
            code2features[code]["response_avg_chars"] += len(text)

        code2features[code]["prompt_avg_chars"] /= code2features[code]["#user_prompts"]
        code2features[code]["response_avg_chars"] /= code2features[code]["#user_prompts"]
```

######	2.2.1. Initialization
	We initialize a defaultdict of defaultdict named code2features to store 			features for each code. This data structure is used to store counts related to user 		prompts and ChatGPT responses.

######	2.2.2. Iterating Through Codes and Conversations:
	Then it iterates through each code and its corresponding conversations (convs) in the 		code2convos dictionary.

######	2.2.3. Counting User Prompts:
	For each user prompt in the conversations, it increments the count of user prompts 		(#user_prompts) for the respective code.


######	2.2.4. Counting Keyword Occurrences:
	For each user prompt, we count the occurrences of keywords from the 				keywords2search list using regular expressions.

######	2.2.5. Calculating Average Characters:
	Keep track of the total number of characters in both user prompts 				(prompt_avg_chars) and ChatGPT responses (response_avg_chars). It later calculates the 		average characters for both.

######	2.2.6. Printing Codes with No Conversations:
	If there are no conversations (convs) for a particular code, it prints the code to 		the console.

######	2.2.7. Normalization of Average Characters:
	Finally we normalize the average characters for user prompts and ChatGPT responses by 		dividing the total characters by the number of user prompts.



#### 2.3.
Then we create a Pandas DataFrame (df) from the feature information stored in the code2features dictionary. 
```
df = pd.DataFrame(code2features).T
df.head(5)
```

#### 2.4. 
Next, we read a CSV file named "scores.csv" and store the resulting DataFrame in the variable named scores. Then some information about scores is displayed. 
```
scores = pd.read_csv("/content/scores.csv", sep=",")
scores["code"] = scores["code"].apply(lambda x: x.strip())

# selecting the columns we need and we care
scores = scores[["code", "grade"]]

# show some examples
scores.head()
```

#### 2.5. 
After that, we modify the structure of the Pandas DataFrame df that was created in a previous section. Reset_index method is used to reset the index of the DataFrame df. The  rename method is used to rename the column with the label "index" to "code". 
```
df.reset_index(inplace=True, drop=False)
df.rename(columns={"index": "code"}, inplace=True)
df.head()
```

#### 2.6. 
Then, we perform a left merge between two Pandas DataFrames (df and question_mapping_scores)
question_mapping_scores) is a dataframe that provides a mapping between codes and their respective similarity scores for each question, it's created earlier.
```
df = pd.merge(df, question_mapping_scores, on="code", how="left")
```

#### 2.7. 
Next, we extend the feature merging process by incorporating information from another DataFrame named scores. The result of the merge and data cleaning operations is stored back in the temp_df. 
```
# Merge the Features
temp_df = pd.merge(df, scores, on='code', how="left")
temp_df.dropna(inplace=True)
temp_df.drop_duplicates("code",inplace=True, keep="first")
temp_df.head()
```

#### 2.8. 
Then we set up the features (X) and the target variable (y) for a machine learning model.
```
# Set the features and target variables
X = temp_df[temp_df.columns[1:-1]]
y = temp_df["grade"]
print(X.shape, y.shape)
```
###### 2.8.1. Feature Selection
temp_df.columns[1:-1] selects all columns from the second column to the second-to-last column of the DataFrame (temp_df). These columns are considered as 			features for the machine learning model.
###### 2.8.2. Target Variable Selection
temp_df["grade"] selects the "grade" column from the DataFrame (temp_df). This column is considered as the target variable for the machine learning model.
###### 2.8.3. Setting Up Features and Target Variable
X is assigned the selected features.
y is assigned the selected target variable.


------------
------------
------------
## 3 - Apply feature subset selection algorithm
For this, releif algorithm is chosen as the feature subset selection.	

#### 3.1.

Here we use scikit-learn's train_test_split function to split the dataset into training and testing sets.



