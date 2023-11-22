# Lip Reading Interface

## Introduction
This interface is used for lip reading classification and supports 1000 Chinese phrases (see [word_index.py](word_index.py)). The interface is used to support the lip reading classification function of a voiceprint recognition system, which ensures the consistency of audio and video content.

## Usage
1. Make sure Python and related dependency libraries are installed.
2. Install Flask and gunicorn:
   ```
   pip install flask gunicorn
   ```
3. Start the service:
   ```
   bash start.sh
   ```
4. Send a POST request to the interface address `http://127.0.0.1:9785/lip-reading` and pass the following parameters:
   - `raw_video_path`: Video file path
   - `start`: Start time (in seconds)
   - `end`: End time (in seconds)
   - `sid`: Identifier
5. Stop the service:
   ```
   bash stop.sh
   ```

## Test Script Usage
1. Modify the parameters in the `test.py` file:
   - Replace `/path/to/your/video.mp4` with the path of the video file you want to test.
   - Set the appropriate start and end times.
   - Replace `your_sid` with your own identifier.
2. Run the test script in the command line:
   ```
   python test.py
   ```
3. Check the returned results.

Note: Make sure the service is started before running the test script.

## Notes
- Make sure the video file is placed in the correct path and the correct file name is used.
- Make sure that the port number 9785 is not occupied by other processes before starting the service.
- Provide the correct parameter information when sending requests.

If you have any questions or concerns, please contact our technical support team.