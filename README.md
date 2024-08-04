# Building Your First Multimodal Gen AI App

## Introduction

Welcome to the tutorial on building your first multimodal generative AI (Gen AI) app! This repository contains all the resources and code you need to get started with creating an app that can generate text, audio, images, and videos using various AI models and APIs.

## Prerequisites

Before you begin, make sure you have the following:

- A GitHub account
- Access to GitHub Codespaces
- API keys for OpenAI, Replicate, and Hugging Face
- Basic knowledge of Python and Bash

## Setting Up the Environment

### Creating a GitHub Codespace

1. Open the repository in GitHub.
2. Click on the `Code` button and select `Create codespace on main`.
3. Wait for the Codespace to spin up.

### Adding Environment Variables

1. In the `.multimodal-app/.streamlit` directory, open the `secrets.toml` file.
2. Add your API keys for OpenAI, Replicate, and Hugging Face as follows:

    ```toml
    openai_api_key = "your_openai_api_key"
    replicate_api_key = "your_replicate_api_key"
    huggingface_api_key = "your_huggingface_api_key"
    ```

3. Ensure these keys are kept private and secure.

**Note for Learners:** If you are taking this workshop at a conference or other event, please check with your instructors or teachers to see if they are providing the API keys for you.

### Creating a Poetry Environment

1. In the Codespace terminal, run:

    ```bash
    poetry shell
    ```

## Application Overview

This multimodal Gen AI app allows you to interact with various AI models to generate text, audio, images, and videos. You can provide inputs via text or voice, and the app will use the appropriate models to create the desired outputs.

## Running the Application

To run the Streamlit app:

1. In the Codespace terminal, navigate to the project directory.
2. Run the following command:

    ```bash
    streamlit run app.py
    ```

The app will open in a new tab in your browser.

## Demo Video

Watch the demo video to see how to set up and use the application:


https://github.com/user-attachments/assets/45da75d3-8a1a-40df-a62d-bcc6e3518240



