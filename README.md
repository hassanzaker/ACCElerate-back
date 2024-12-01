
# Back-End for Hockey Video Analyzer

This is the back-end component of the Hockey Video Analyzer project. The back-end processes uploaded videos to analyze player movements and generates insights such as player-specific plots. It is built with Flask and provides RESTful APIs for interaction with the front-end.

## Features

- Upload video files or provide video URLs for analysis.
- Cache processed videos to avoid redundant computations.
- Generate and serve player-specific plots.
- Supports MP4, AVI, and MOV file formats.

## Requirements

- Python 3.8 or higher
- Flask
- Flask-CORS
- Werkzeug
- Requests
- Matplotlib
- FFmpeg

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/hockey-analyzer-backend.git
   cd hockey-analyzer-backend
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate   # For Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Install FFmpeg:

   Follow the instructions on [FFmpeg's official website](https://ffmpeg.org/download.html) to install FFmpeg.

## Usage

1. Start the Flask server:

   ```bash
   python run.py
   ```

2. The server will be available at `http://127.0.0.1:5000`.

3. Use the following endpoints:
   - **POST /upload**: Upload a video file or provide a video URL for processing.
   - **GET /player_ids**: Retrieve a list of player IDs from the processed data.
   - **GET /plot/<player_id>**: Generate and retrieve a plot for a specific player.

## Docker

You can also run the back-end in a Docker container.

1. Build the Docker image:

   ```bash
   docker build -t hockey-analyzer-backend .
   ```

2. Run the Docker container:

   ```bash
   docker run -p 5000:5000 hockey-analyzer-backend
   ```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.
