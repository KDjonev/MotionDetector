#!/usr/bin/python

import cv2


def ScoreFrameForDetection(
  Frame,
  BackgroundSubtractionKernel,
  BackgroundSubtractor):

    BackgroundSubtractionScore = ScoreByBackgroundSubtraction(
        Frame,
        BackgroundSubtractionKernel,
        BackgroundSubtractor)

    return BackgroundSubtractionScore


def ScoreByBackgroundSubtraction(
  Frame,
  BackgroundSubtractionKernel,
  BackgroundSubtractor):

    ForegroundMaskedImage = BackgroundSubtractor.apply(Frame)
    ForegroundMaskedImage = cv2.morphologyEx(
        ForegroundMaskedImage,
        cv2.MORPH_OPEN,
        BackgroundSubtractionKernel)
    ForegroundPixels = cv2.findNonZero(ForegroundMaskedImage)

    if (ForegroundPixels is None):
        ForegroundPixelCount = 0
    else:
        ForegroundPixelCount = len(ForegroundPixels)

    # The score is just the number of pixels detected as foreground
    return ForegroundPixelCount


def DetectMotionFromWebcam(Threshhold, ConsecutiveToConsider):
    # Returns the Image that caused the Score to be average score to be greater
    # than a certain threashold. The score is decided from the webcam based
    # on a combination of scoring functions. Once a score above a threshold is
    # detected, we consider the next ConsecutiveToConsider frames as well and
    # average the accumulated score to determine the true score

    AccummulatedFrameScore = 0.0
    CurrentFrameScore = 0

    Webcam = cv2.VideoCapture(0)

    BackgroundSubtractionKernel = \
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    BackgroundSubtractor = cv2.bgsegm.createBackgroundSubtractorGMG()

    while(1):
        ReturnValue, Frame = Webcam.read()
        if (ReturnValue != 0):
            CurrentFrameScore = ScoreFrameForDetection(
                Frame,
                BackgroundSubtractionKernel,
                BackgroundSubtractor)
            if (CurrentFrameScore > Threshhold):
                for Count in range(ConsecutiveToConsider):
                    ReturnValue, Frame = Webcam.read()
                    if (ReturnValue != 0):
                        AccummulatedFrameScore += ScoreFrameForDetection(
                            Frame,
                            BackgroundSubtractionKernel,
                            BackgroundSubtractor)
                AccummulatedFrameScore /= ConsecutiveToConsider

            if (AccummulatedFrameScore > Threshhold):
                return Frame
            else:
                AccummulatedFrameScore = 0

    Webcam.release
    return None


if __name__ == '__main__':

    from time import gmtime, strftime, sleep
    import sys

    sys.path.append("../SendEmail")
    from EmailOptions import EmailOptions
    import EmailSender

    Image = DetectMotionFromWebcam(15000, 50)
    try:
        while (Image is not None):
            cv2.imwrite("MotionDetectedImage.png", Image)

            EmailOptions = EmailOptions()
            EmailOptions.LoadOptionsFromXml("HaltEmailOptions.eo")
            EmailOptions.ParseOptionsFromCommandLine()
            EmailSender.SendEmail(
                EmailOptions.FromAddress,
                EmailOptions.ToAddresses,
                strftime("%Y-%m-%d %H:%M:%S ", gmtime()) + EmailOptions.Subject,
                EmailOptions.Body,
                "*.png",
                EmailOptions.FromAddressPassword)

            # Wait 10 minutes before trying again
            sleep(600)
            Image = DetectMotionFromWebcam(15000, 50)

    except KeyboardInterrupt:
        sys.exit()
