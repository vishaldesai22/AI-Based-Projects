import cv2

# Load the trained weapon (gun) cascade classifier
gun_cascade = cv2.CascadeClassifier('cascade.xml')

# Open the default camera using DirectShow (fixes Windows freeze issue)
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check if the camera opened successfully
if not camera.isOpened():
    print("❌ Cannot open camera")
    exit()

while True:
    # Read a frame from the camera
    ret, frame = camera.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Resize frame without imutils
    frame = cv2.resize(frame, (600, 400))

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect guns in the grayscale image
    guns = gun_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))

    # Draw rectangles around detected guns
    for (x, y, w, h) in guns:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Gun Detected", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Show the processed video frame
    cv2.imshow("Weapon Detection - Press Q to Exit", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all OpenCV windows
camera.release()
cv2.destroyAllWindows()
