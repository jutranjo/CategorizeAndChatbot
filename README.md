# Message Categorization and Chatbot Assistant

 This project is built to analyze, categorize, and interactively explore user feedback via a CLI chatbot.

It is built to process records in a dataset that contains:
- `id_user`: A unique user identifier
- `timestamp`: When the message was received
- `source`: Platform of origin (`livechat` or `telegram`)
- `message`: Free-text input from the user 


## Setup Instructions

### 1. Clone the Repository

```bash
git clone git@github.com:jutranjo/CategorizeAndChatbot.git
cd CategorizeAndChatbot
```
---
### 2. Create and Activate a Virtual Environment

Create a new Python virtual environment (recommended):

```bash
# On macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# On Windows
python -m venv .venv
.\.venv\Scripts\activate
```
---

### 3. Install Required Packages

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```
---

### 4. Set Up the `.env` File

Create the `.env` file in the root directory and add your OpenAI API key:

```ini
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```
---

## Project Structure

```
.
├── Chatbot/
│   ├── chatbot.py                # Main script to run the chatbot
│   ├── stats.py                  # Data summaries and print utilities
│   └── test_chatbot.py           # Pytest suite for chatbot behavior
│
├── categorization/
│   ├── visualize_UMAP.py              # Visualization of clustering via UMAP + MatPlotLib
│   ├── zero_shot_classification.py    # Zero-shot classification if labels are known beforehand
│   ├── KMeans_category_clustering.py  # Zero-shot semantic clustering via MiniLM + KMeans
│   ├── name_categories.py             # Script to inspect clusters and assign human-readable labels
│   └── merge_categories.py            # Merge numeric labels with names to produce the labeled CSV
│
├── merged_messages_with_categories.csv   # Final labeled dataset used by the chatbot
├── requirements.txt
├── LLM-DataScientist-Task_Data.csv
└── .env
```
---

## Categorization Pipeline

You must run the scripts in the following order to produce the labeled dataset:

### 1. **KMeans Clustering**
```bash
python categorization/KMeans_category_clustering.py
```
- Loads `LLM-DataScientist-Task_Data.csv`
- Embeds each message using `all-MiniLM-L6-v2`
- Applies KMeans clustering (variable number of clusters)
- Saves a `clustered_messages.csv`

### 2. **Name Each Category**
```bash
python categorization/name_categories.py
```

- Loads `clustered_messages.csv`
- Prompts you with example messages from each cluster
- Lets you assign a label to each cluster
- Saves a `cluster_category_mapping.csv`
### 3. **Merge Categories into Final Dataset**
```bash
python categorization/merge_categories.py
```

- Loads `cluster_category_mapping.csv` and `clustered_messages.csv`
- Merges labeled clusters into `merged_messages_with_categories.csv`
- This file is required by the chatbot and must be in the top most project directory

---
Alternatively, if you know exactly what categories you want the messages sorted in, you can edit `zero_shot_classification.py` with the correct labels, then run that.
```bash
python categorization/zero_shot_classification.py
```

- Loads `LLM-DataScientist-Task_Data.csv`
- Saves labeled messages into `merged_messages_with_categories.csv`
- This file is required by the chatbot and must be in the top most project directory

---
## Running the Chatbot

Once you have your `.env` set up and labeled messages ready:

```bash
python -m Chatbot.chatbot
```

You can now ask questions like:
- “Show me freespin issues in the last 7 days”
- “Now just show Telegram messages”
- “Reset”
- “Only deposit issues from yesterday”

---

## Testing

To run all tests:
```bash
pytest Chatbot/test_chatbot.py
```

The test suite validates:
- Filter extraction accuracy
- Time range interpretation
- Conversational refinement vs. reset behavior

---
## Evaluation Questions.

### How did you classify feedback (Describe your methodology: rule-based, machine learning, hybrid. Discuss advantages, drawbacks, and how you would handle new, previously unseen issues, such as a new wallet blocking deposits).
 I used machine learning, first visualizing with UMAP to get a rough estimate of the number of clusters, determined that to be 10, then used KMeans clustering using a zero shot approach utilizing the sentence and short paragraph encoder `all-MiniLM-L6-v2` to categorize each message into one of 10 distinct categories, grouping messages with similar content together, that is, assigning them the same label integer. I then used a script to print out examples of each category and gave each category a string label by hand.

### How does your chatbot manage conversational context?
 A system prompt is used to have the LLM interpret the user request into a dictionary of filters. No conversational context beyond the system prompt and current query is needed in this case as all the relevant messages written beforehand are stored locally and merged with previous filters as needed.

### Q: What are the main limitations? (e.g., vague feedback, multi-category overlaps, conversational memory constraints.)
 Messages are categorized into singular categories, so if a user message had multiple issues it would fall into only one of them and potentially not alerting the appropriate CS team member.

 If the wording is ambigious on the requested timeframe (such as Telegram messages in the last day) it is possible the chatbot will give the a filter for the timeframe that is different than what the user expects (last day might mean 24 hours or it might mean since the last midnight or since the start of the workday).
As the messages end up being transformed into active filters there is no constraint on how many times the filter can be revised. A limitation is that the chatbot is unable to respond to requests such as "Show me the telegram messages from before again" but would have to be instructed again "Show me telegram messages about login issues in the past 3 days".

### Q: How could the system be improved?
 With domain knowledge of all kinds of issues users are reporting or the knowledge of which teams have actionable responses to user issues, the categories could be written by hand beforehand, and then messages assigned to each category. A better sentence encoder than 'all-MiniLM-L6-v2' could be used, such as `gemini-embedding-exp-03-07` or `Ling-Embed-Mistral`. The initial categorization of messages could also be made more automated but as it isn't user facing I did not prioritize this.

### Q: Explain how the chatbot tracks and utilizes past queries to refine current requests.
 It stores the filters used in the previous query and gauges when the user wants a new filtered search unrelated to the current one. As a failsafe the user can type 'reset' to remove all filters currently in use. If filters are not reset via the 'reset' command and the LLM determines the user wants to refine the search, it will supply new filters for the fields the user wants filtered while leaving the other fields intact. 

### Q: If the entire conversation were provided (not just a single response, but a full exchange with a support agent), would you approach this task differently? (Explain how.)
 Without domain knowledge of the issues customer support is helping users with, I would perform a similar approach of examining clusters in the semantic similarity of conversations, then figuring out what the categories are. It is probable whole conversations would result in additional categories as compared to just each individual message as the conversation with the CS representative could be classified as 'Successfully resolved', 'Unsuccesfully resolved', and 'Other', resulting in more categories and potentially more ambiguity in which category which conversation fits. 

 The other parts of the chatbot would likely not need much modification, as it's written to function with a variable number of messages in the database in some number of categories.

### Q: How would you measure and validate the correctness of message classifications?
 A sample of the messages would have to be manually labeled. Once labeled, I would use various metrics to determine the accuracy, precision, recall and F1 score of the sentence encoder. 

 If needed, to gauge how ambigious the user messages were, multiple annotators could be used and then Cohen's kappa score could be computed to determine the level of agreement between different human annotators. 