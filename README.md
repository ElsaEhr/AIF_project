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

A compléter  
```
/project_AIF
│
├─ /part1                
├─ /part2                
├─ /part3                
├─ /part4               
├─ Dockerfile
├─ requirements.txt
└─ README.md
```

## Installation

1. Clone the repository:

```bash
git clone <repo_url>
cd <repo_name>
```

A modifier

2. Build the Docker image:

```bash
docker build -t movie-ai-platform .
```

3. Run the container locally:

```bash
docker run -p 7860:7860 movie-ai-platform
```

4. Access the web interface at `http://localhost:7860`.

## Usage

The following functionalities are accessible via the **Gradio web interface** powered by the **Flask REST API**:

| Functionality | Input | Output | Part Introduced |
| :--- | :--- | :--- | :--- |
| **Poster Genre Prediction** | Uploaded Poster Image | Predicted Genre | Part 1 |
| **Poster Validation** | Uploaded Image | Validation Status (Valid/Invalid) | Part 2 |
| **Plot Genre Prediction** | Movie Plot Text | Predicted Genre | Part 3 |
| **Plot Recommendations** | Movie Plot Text | List of Similar Movies | Part 3 |
| **Conversational Search** | Natural Language Query (e.g., "Find funny movies from the 90s") | Relevant Movie Suggestions | Part 4 |

## Deployment
Every part of this project adheres to a continuous deployment cycle, involving the implementation of new features via a Flask REST API, their integration into an evolving Gradio web interface, and the final containerization using Docker before being tested and deployed to a cloud provider.

## Dataset & Model

* **Movie Posters Dataset:** https://drive.google.com/file/d/1-1OSGlN2EOqyZuehBgpgI8FNOtK-caYf/view
%* **Plot Dataset:** [link]
* **Pretrained models:** Download at runtime via Docker configuration.


