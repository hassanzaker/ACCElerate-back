from flask import Flask, request, jsonify, send_file
import os, io
from werkzeug.utils import secure_filename
from Analyzer import process_video, get_user_ids, get_player_keypoints, plot_player_data
from flask_cors import CORS
import ffmpeg
import requests



def create_app():
    app = Flask(__name__)

    processed_data_cache = {}



    # Enable CORS for all routes
    CORS(app)



    # Configuration
    UPLOAD_FOLDER = 'uploads'
    OUTPUT_FOLDER = 'processed_videos'
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

    # Ensure folders exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    @app.route('/upload', methods=['POST'])
    def upload_video():
        input_path = None  # Initialize the input path variable

        # Check if a file was uploaded
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(input_path)

        # Check if a URL was provided
        elif 'url' in request.form:
            video_url = request.form['url']
            if not video_url:
                return jsonify({"error": "No video URL provided"}), 400

            # Download the video from the URL
            try:
                response = requests.get(video_url, stream=True)
                if response.status_code == 200:
                    filename = video_url.split("/")[-1]
                    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    with open(input_path, 'wb') as video_file:
                        for chunk in response.iter_content(chunk_size=1024):
                            video_file.write(chunk)
                else:
                    return jsonify({"error": "Failed to download video from URL"}), 400
            except Exception as e:
                return jsonify({"error": f"Error downloading video: {str(e)}"}), 500

        else:
            return jsonify({"error": "No file or URL provided"}), 400

        # Generate output filenames
        output_filename = f"processed_{filename}"
        avi_output_filename = output_filename.replace(".mp4", ".avi")
        avi_output_path = os.path.join(app.config['OUTPUT_FOLDER'], avi_output_filename)

        mp4_output_filename = avi_output_filename.replace(".avi", ".mp4")
        mp4_output_path = os.path.join(app.config['OUTPUT_FOLDER'], mp4_output_filename)

        # Check if processed file already exists (cache system)
        if os.path.exists(mp4_output_path):
            print("Cache hit: Processed file already exists.")
            return send_file(mp4_output_path, as_attachment=True), 200

        # Process the video if no cached file is found
        try:
            tracks, keypoints = process_video(input_path, avi_output_path)
            processed_data_cache[filename] = {'tracks': tracks, 'keypoints': keypoints}

        except Exception as e:
            return jsonify({"error": f"Error processing video: {str(e)}"}), 500

        # Convert AVI to MP4
        converted_file_path = convert_avi_to_mp4(avi_output_path, mp4_output_path)
        if not converted_file_path:
            return jsonify({"error": "Failed to convert AVI to MP4"}), 500

        # Send the MP4 file to the front-end
        return send_file(converted_file_path, as_attachment=True), 200

    @app.route('/plot/<int:player_id>', methods=['GET'])
    def plot_data(player_id):
        try:
            # Check if there are processed tracks available in the cache
            if not processed_data_cache:
                return jsonify({"error": "No processed data available. Upload a video first."}), 400

            # Get the last processed video's tracks and keypoints
            last_processed_data = list(processed_data_cache.values())[-1]
            tracks = last_processed_data.get('tracks', None)
            keypoints = last_processed_data.get('keypoints', None)

            print("TTTT", tracks, keypoints)

            if not tracks or not keypoints:
                return jsonify({"error": "Tracks or keypoints not available."}), 500

            player_ids = get_user_ids(tracks)
            # Convert all player IDs to standard Python integers
            player_ids = [int(player_id) for player_id in player_ids]

            # Generate player data and plot
            player_data = get_player_keypoints(keypoints, tracks, player_id)
            fig = plot_player_data(player_data, player_ids[0])

            # Save the plot to a BytesIO object and send it as a response
            buf = io.BytesIO()
            fig.savefig(buf, format='png')  # Save the figure to the buffer
            buf.seek(0)  # Rewind the buffer to the beginning
            return send_file(buf, mimetype='image/png')  # Send the image as a response

        except Exception as e:
            # Log the error
            print(f"Error generating plot for player ID {player_id}: {e}")
            return jsonify({"error": f"Error generating plot: {str(e)}"}), 500
            

    @app.route('/player_ids', methods=['GET'])
    def get_player_ids():
        # Check if there are processed tracks available in the cache
        if not processed_data_cache:
            return jsonify({"error": "No processed data available. Upload a video first."}), 400

        # Get the last processed video's tracks
        last_processed_data = list(processed_data_cache.values())[-1]
        tracks = last_processed_data.get('tracks', None)

        if tracks:
            player_ids = get_user_ids(tracks)

            # Convert all player IDs to standard Python integers
            player_ids = [int(player_id) for player_id in player_ids]


            return jsonify({"player_ids": player_ids}), 200
        else:
            return jsonify({"error": "Tracks not available in the processed data."}), 500

    # Add routes here
    @app.route('/')
    def home():
        return "Welcome to the Flask App!"
    

    return app

    

def convert_avi_to_mp4(input_file, output_file):
    """
    Converts an AVI file to MP4 format using FFmpeg.

    Parameters:
        input_file (str): Path to the input .avi file.
        output_file (str): Path to save the converted .mp4 file.

    Returns:
        str: Path to the converted .mp4 file.
    """
    try:
        ffmpeg.input(input_file).output(output_file, vcodec='libx264', acodec='aac', strict='experimental').run(overwrite_output=True)
        print(f"Conversion successful! MP4 file saved to: {output_file}")
        return output_file
    except ffmpeg.Error as e:
        print(f"An error occurred: {e.stderr.decode()}")
        return None