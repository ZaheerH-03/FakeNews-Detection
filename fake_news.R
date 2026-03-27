# =============================================================================
# Fake News Detection using NLP and Machine Learning in R
# =============================================================================
# Author      : [Zaheer Hussain]
# Dataset     : WELFake Dataset (https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
# Description : This script performs EDA, text preprocessing, and trains
#               Naive Bayes and Random Forest classifiers to detect fake news.
# Models      : Naive Bayes, Random Forest
# Language    : R
# =============================================================================

# ------------------------------------------------------------------------------
# 1. Load Libraries
# ------------------------------------------------------------------------------
library(ggplot2)
library(dplyr)
library(tidyverse)
library(stringr)
library(caTools)

# Text Mining
library(tm)
library(wordcloud)
library(RColorBrewer)
library(text2vec)
library(Matrix)

# Modelling
library(e1071)         # Naive Bayes
library(caret)         # Evaluation metrics & splitting
library(randomForest)  # Random Forest


# ------------------------------------------------------------------------------
# 2. Load and Inspect Dataset
# ------------------------------------------------------------------------------
data <- read.csv("WELFake_Dataset.csv")
df   <- as.data.frame(data)

cat("=== Dataset Summary ===\n")
print(summary(df))
cat("Dimensions:", nrow(df), "rows x", ncol(df), "columns\n\n")

# Label distribution
per_real <- (sum(df$label == 1) / nrow(df)) * 100
per_fake <- (sum(df$label == 0) / nrow(df)) * 100
cat(sprintf("Real news: %.2f%%\n", per_real))
cat(sprintf("Fake news: %.2f%%\n\n", per_fake))


# ------------------------------------------------------------------------------
# 3. Data Cleaning
# ------------------------------------------------------------------------------

# Remove unnamed index columns (e.g., added by Python/Excel exports)
df <- df[, !grepl("^Unnamed|^X", names(df))]

# Count and visualise NA / empty values per column
count_na <- sapply(df, function(col) sum(is.na(col) | col == ""))
na_df    <- data.frame(Column = names(count_na), NA_Count = count_na)

cat("=== Missing / Empty Values per Column ===\n")
print(na_df)

ggplot(na_df, aes(x = Column, y = NA_Count)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Missing Values per Column", x = "Column", y = "Count") +
  theme_minimal()

# Drop rows that have any empty string in any column
df <- df[!apply(df, 1, function(row) any(row == "")), ]
cat(sprintf("Rows after removing empties: %d\n\n", nrow(df)))


# ------------------------------------------------------------------------------
# 4. Feature Engineering
# ------------------------------------------------------------------------------

# Combined title + text field and its character length (excluding spaces)
df[["title_text"]] <- paste(df[["title"]], df[["text"]])
df[["length"]]     <- nchar(df[["title_text"]]) - str_count(df[["title_text"]], " ")


# ------------------------------------------------------------------------------
# 5. Exploratory Data Analysis (EDA)
# ------------------------------------------------------------------------------

# --- 5a. Label distribution bar chart ---
ggplot(df, aes(x = factor(label), fill = factor(label))) +
  geom_bar() +
  scale_fill_manual(values = c("0" = "#FF5733", "1" = "#33FFB8"),
                    labels = c("0" = "Fake", "1" = "Real")) +
  scale_x_discrete(labels = c("0" = "Fake", "1" = "Real")) +
  labs(title = "Fake vs Real News Distribution", x = "Label", y = "Count", fill = "Label") +
  theme_minimal()

# --- 5b. Body length distribution (full range) ---
ggplot(df, aes(x = length, fill = factor(label))) +
  geom_histogram(binwidth = 500, alpha = 0.7, position = "identity") +
  scale_fill_manual(values = c("0" = "#FF5733", "1" = "#33FFB8"),
                    labels = c("0" = "Fake", "1" = "Real")) +
  labs(title = "Body Length Distribution by Label (Full Range)",
       x = "Character Length", y = "Frequency", fill = "Label") +
  theme_minimal() +
  theme(legend.position = "top")

# --- 5c. Body length distribution (zoomed in) ---
ggplot(df, aes(x = length, fill = factor(label))) +
  geom_histogram(binwidth = 4, alpha = 0.7, position = "identity") +
  scale_fill_manual(values = c("0" = "#FF5733", "1" = "#33FFB8"),
                    labels = c("0" = "Fake", "1" = "Real")) +
  labs(title = "Body Length Distribution by Label (Zoomed: 0–200)",
       x = "Character Length", y = "Frequency", fill = "Label") +
  scale_x_continuous(limits = c(0, 200), breaks = seq(0, 200, by = 40)) +
  scale_y_continuous(limits = c(0, 50),  breaks = seq(0, 50,  by = 10)) +
  theme_minimal() +
  theme(legend.position = "top")


# ------------------------------------------------------------------------------
# 6. Train / Test Split (80 / 20)
# ------------------------------------------------------------------------------
set.seed(53)

X     <- df$text
y     <- df$label
split <- sample.split(y, SplitRatio = 0.80)

X_train <- subset(X, split == TRUE)
X_test  <- subset(X, split == FALSE)
y_train <- subset(y, split == TRUE)
y_test  <- subset(y, split == FALSE)

cat(sprintf("Training samples : %d\n", length(X_train)))
cat(sprintf("Test samples     : %d\n\n", length(X_test)))


# ------------------------------------------------------------------------------
# 7. Word Clouds
# ------------------------------------------------------------------------------

# --- 7a. All titles ---
titles <- paste(df$title, collapse = " ")
wordcloud(titles, max.words = 300, random.order = FALSE,
          colors = brewer.pal(8, "Dark2"), scale = c(3, 0.5), min.freq = 2)
title("Word Cloud – All Titles")

# --- 7b. Fake news body text ---
#fake_texts <- paste(X_train[y_train == 0], collapse = " ")
#wordcloud(fake_texts, max.words = 300, random.order = FALSE,
#          colors = brewer.pal(8, "Reds"), scale = c(3, 0.5), min.freq = 2)
#title("Word Cloud – Fake News")

# --- 7c. Real news body text ---
#real_texts <- paste(X_train[y_train == 1], collapse = " ")
#wordcloud(real_texts, max.words = 300, random.order = FALSE,
#          colors = brewer.pal(8, "Greens"), scale = c(3, 0.5), min.freq = 2)
#title("Word Cloud – Real News")


# ------------------------------------------------------------------------------
# 8. Text Preprocessing  (Bag-of-Words)
# ------------------------------------------------------------------------------

preprocess_corpus <- function(text_vector) {
  corpus <- Corpus(VectorSource(text_vector))
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, removeWords, stopwords("SMART"))
  corpus <- tm_map(corpus, stripWhitespace)
  return(corpus)
}

# Build training DTM and remove sparse terms (keep top ~10% most frequent)
corpus_train    <- preprocess_corpus(X_train)
dtm_train       <- DocumentTermMatrix(corpus_train)
dtm_train_sparse <- removeSparseTerms(dtm_train, sparse = 0.90)

cat(sprintf("Vocabulary size after sparsity filter: %d terms\n\n",
            ncol(dtm_train_sparse)))

# Convert to data frames
train_matrix <- as.matrix(dtm_train_sparse)
train_df     <- as.data.frame(train_matrix)
train_df$label <- factor(y_train)

# Align test DTM to training vocabulary (avoids unseen-term issues)
corpus_test      <- preprocess_corpus(X_test)
dtm_test         <- DocumentTermMatrix(corpus_test)
dtm_test_aligned <- dtm_test[, Terms(dtm_train_sparse)]

test_matrix <- as.matrix(dtm_test_aligned)
test_df     <- as.data.frame(test_matrix)
test_df$label <- factor(y_test)

# Sanitise column names (required by randomForest)
sanitise_colnames <- function(df) {
  colnames(df) <- make.names(colnames(df), unique = TRUE)
  colnames(df) <- gsub("[^a-zA-Z0-9_]", "", colnames(df))
  return(df)
}

train_df <- sanitise_colnames(train_df)
test_df  <- sanitise_colnames(test_df)

# Keep only columns common to both splits (label included)
common_cols <- intersect(colnames(train_df), colnames(test_df))
train_df    <- train_df[, common_cols]
test_df     <- test_df[, common_cols]

# Restore label as factor after column alignment
train_df$label <- factor(y_train)
test_df$label  <- factor(y_test)


# ------------------------------------------------------------------------------
# 9. Model 1 – Naive Bayes
# ------------------------------------------------------------------------------
cat("=== Training Naive Bayes ===\n")
nb_model     <- naiveBayes(label ~ ., data = train_df)
nb_preds     <- predict(nb_model, test_df)
nb_cm        <- confusionMatrix(nb_preds, test_df$label)

print(nb_cm)
cat(sprintf("Naive Bayes Accuracy: %.2f%%\n\n",
            nb_cm$overall["Accuracy"] * 100))


# ------------------------------------------------------------------------------
# 10. Model 2 – Random Forest
# ------------------------------------------------------------------------------
cat("=== Training Random Forest (ntree = 300) ===\n")
set.seed(123)
rf_model  <- randomForest(label ~ ., data = train_df, ntree = 300)
print(rf_model)

rf_preds  <- predict(rf_model, test_df)
rf_cm     <- confusionMatrix(rf_preds, test_df$label)

print(rf_cm)
cat(sprintf("Random Forest Accuracy: %.2f%%\n\n",
            rf_cm$overall["Accuracy"] * 100))


# ------------------------------------------------------------------------------
# 11. Model Comparison Summary
# ------------------------------------------------------------------------------
comparison <- data.frame(
  Model    = c("Naive Bayes", "Random Forest"),
  Accuracy = c(
    round(nb_cm$overall["Accuracy"] * 100, 2),
    round(rf_cm$overall["Accuracy"] * 100, 2)
  ),
  Precision = c(
    round(nb_cm$byClass["Pos Pred Value"] * 100, 2),
    round(rf_cm$byClass["Pos Pred Value"] * 100, 2)
  ),
  Recall = c(
    round(nb_cm$byClass["Sensitivity"] * 100, 2),
    round(rf_cm$byClass["Sensitivity"] * 100, 2)
  ),
  F1 = c(
    round(nb_cm$byClass["F1"] * 100, 2),
    round(rf_cm$byClass["F1"] * 100, 2)
  )
)

cat("=== Model Comparison ===\n")
print(comparison)

ggplot(comparison, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", width = 0.5) +
  geom_text(aes(label = paste0(Accuracy, "%")), vjust = -0.5, size = 5) +
  scale_fill_manual(values = c("Naive Bayes" = "#4E9AF1", "Random Forest" = "#F4A261")) +
  labs(title = "Model Accuracy Comparison", y = "Accuracy (%)", x = "") +
  ylim(0, 105) +
  theme_minimal() +
  theme(legend.position = "none")