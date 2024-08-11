# Building Your First Multimodal Gen AI App 🚀

## Introduction

Welcome to the tutorial on building your first multimodal generative AI (Gen AI) app! This repository contains all the resources and code you need to get started with creating an app that can generate text, audio, images, and videos using various AI models and APIs.

## Prerequisites

Before you begin, make sure you have the following:

- A GitHub account
- GitHub Codespaces enabled (comes with your GitHub account)
- API keys for the following:
  - [Groq](https://groq.com/) or [OpenAI](https://platform.openai.com/playground) (at least one is required; Groq has a free tier for all the models we need!)
  - [Replicate](https://replicate.com/) (necessary for full functionality; Replicate has kindly provided credits for those taking this workshop at a conference.)
- Basic knowledge of Python and Bash

**Note about GitHub Codespaces:** 
- GitHub Codespaces is included with every GitHub account.
- There's a substantial monthly free tier for personal accounts (120 core hours/month as of 2024).
- If you exceed the free tier, you may need to purchase additional usage.
- For the latest information on GitHub Codespaces pricing and usage limits, please check the [official GitHub documentation](https://docs.github.com/en/billing/managing-billing-for-github-codespaces/about-billing-for-github-codespaces).

**Note for Workshop Participants:** If you are taking this workshop at a conference or other event, please check with your instructors or teachers to see if they are providing the API keys for you.

## Setting Up the Environment

To get up and running, you can watch the video below and/or follow the instructions:



https://github.com/user-attachments/assets/a311c3fd-628b-4c14-aafa-dfeaf8a52885




### Creating a GitHub Codespace

1. Open the repository in GitHub.
2. Click on the `Code` button and select `Create codespace on main`.
3. Wait for the Codespace to spin up (this should take about 2 minutes).

### Adding API Keys

1. In the Codespace, navigate to the `.streamlit` directory inside the `multimodal_app` folder.
2. Open the `secrets.toml` file.
3. Add your API keys as follows:
    ```toml
    OPENAI_API_KEY = "your_openai_api_key"
    GROQ_API_KEY = "your_groq_api_key"
    REPLICATE_API_TOKEN = "your_replicate_api_token"
    ```
4. Save the file and ensure these keys are kept private and secure.

### API Keys

You'll need API keys for either OpenAI or Groq, and Replicate (for full functionality).

### Setting Up the Poetry Environment

1. Once the Codespace finishes configuring, it will automatically install Poetry.
2. In the Codespace terminal, activate the Poetry environment:
    ```bash
    cd multimodal_app
    poetry shell
    ```

## Running the Application

To run the Streamlit app:

1. Ensure you're in the `multimodal_app` directory and have activated the Poetry shell.
2. Run the following command:
    ```bash
    streamlit run main.py
    ```
3. Click "Open in browser" when prompted to view the app.

## Using the Application

The multimodal Gen AI app allows you to:

1. Record speech or type text input.
2. Transcribe speech to text.
3. Generate text responses based on your input.
4. Create audio versions of the text.
5. Generate images based on the content.
6. Create videos incorporating the generated content.

To use the app:

1. Click the record button to speak, or type your input.
2. Click "Transcribe" to convert speech to text (if applicable).
3. Choose to run all tasks concurrently or step-by-step.
4. Explore the generated text, audio, images, and videos.

## Troubleshooting

If you encounter any issues:

- Ensure all API keys are correctly entered in the `secrets.toml` file.
- Check that you're in the correct directory (`multimodal_app`) when running commands.
- Verify that all dependencies are installed by running `poetry install` if needed.

## Contributing

We welcome contributions to improve this project! Please feel free to submit issues or pull requests.

## License

[Insert appropriate license information here]

---

Happy building! We hope you enjoy creating your first multimodal Gen AI app. If you have any questions or feedback, please don't hesitate to reach out.
