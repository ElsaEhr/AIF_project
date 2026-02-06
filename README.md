# AI Tools for a Movie Streaming Platform

## Project members 
Zoe Lelong (lelong@insa-toulouse.fr)  
Jade Roumazeille-Peter (roumazeille@insa-toulouse.fr)  
Eduardo De Jesus Zancanaro Garcia (zancanaro@insa-toulouse.fr)  
Elsa Ehrhart (ehrhart@insa-toulouse.fr)  
Aria Chauchat (chaucha@insa-toulouse.fr)

## Overview

This project integrates AI tools into a movie streaming platform to enhance user experience and automate catalog management. Users can predict movie genres, validate posters, get recommendations, and discover movies through natural language interaction.

## Features

The application is developed across four main parts, each introducing a critical capability:

* **Part 1 — Genre Prediction from Posters (CNN)**: Automatically predicts a movie's single genre using its poster image via a Convolutional Neural Network (CNN).
* **Part 2 — Movie Poster Validation**: Extends the API to detect and reject invalid image uploads, ensuring catalog quality.
* **Part 3 — Genre & Recommendation from Plots (NLP)**: Uses NLP techniques on the movie plot to predict genres and generate similar movie recommendations (via **Annoy Index**), primarily for cases where a poster is unavailable or invalid.
* **Part 4 — Natural Language Movie Discovery**: Implements a conversational interface allowing users to find movies they might like through natural language queries.

  
  

## Project Structure
```
/AIF_project
│
├─ /main
├─ /part1                
├─ /part2                
├─ /part3                
└─ /part4               
 
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/ElsaEhr/AIF_project.git 
cd AIF_project
```
2. Download the poster database:

Please follow this link to access the poster database and download the folder ```\content``` into the ```\AIF_project``` folder in your local repo: https://drive.google.com/drive/folders/17WB35oYygrQDlcK9XLaKdh-akmBB1lfD?usp=share_link 

Your structure should look like this: 

AIF_project/
├── content/
├── ...


2. Build the Docker image:

In a terminal: 

```bash
docker compose build 
```

3. Run the container locally:

```bash
docker compose up
```

4. Access the web interface at `http://localhost:7860`.


5. Enjoy the application as you like! (See **Usage** for more details)


6. When finish using the application:

```bash
docker compose down
```

## Usage

The following functionalities are accessible via the **Gradio web interface** powered by the **Flask REST API**:

| Functionality | Input | Output | Part Introduced | Web Tab |
| :--- | :--- | :--- | :--- |:--- |
| **Poster Genre Prediction** | Uploaded Poster Image | Predicted Genre | Part 1 | Poster Analyzer |
| **Poster Validation** | Uploaded Poster Image | Validation (Is the image a poster or not ?) | Part 2 | Poster Analyzer |
| **Plot Genre Prediction** | Movie Plot Text | Predicted Genre | Part 3.1 | Plot Tools |
| **Recommendations from plot (NLP)** | Movie Plot Text | List of Similar Movies | Part 3.2 | Plot Tools |
| **Recommendations from plot (Retrieval)** | Movie Plot Text | List of Similar Movies | Part 4.1 | Retrival CLIP + Annoy |
| **Conversational Search** | Natural Language Query (e.g., "I want a movie about pirates") | Relevant Movie Suggestions | Part 4.2 | Movie Discovery Chatbot |

## Deployment
Every part of this project adheres to a continuous deployment cycle, involving the implementation of new features via a Flask REST API, their integration into an evolving Gradio web interface, and the final containerization using Docker before being tested and deployed to a cloud provider.

## Cloud
In order to have access to the application online, please go to this link:  http://34.121.20.146:7860  
(specified time to be determined)
