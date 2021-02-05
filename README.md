## This project is no longer being developed.

Please go to https://github.com/LedFx/LedFx for the latest in funky lighting!

Good news! **LedFx is compatable with WLED, and the ESP firmware used by Systematic LEDs AND Scott Lawson's audio-reactive-led-strip.**

LedFx couldn't be simpler to get running if you're a WLED user.

1. Download and install from https://ledfx.app 
2. Run LedFx and your WLED devices will be detected automatically, ready for you to use!

All you have to do is download and install LedFx, add the ESP device(s), and you're good to go :D

## Systematic LEDs

All in one lighting solution for your computer. Mood lighting? Club effects?

Lighting your room or office should be a simple procedure, with flexibilty to transition between moods. 

Grab some LED strips, and you can set this up yourself. Follow the simple steps outlines below to tune your space to a mood that fits you.

## Installation

Check out [this tutorial](https://www.youtube.com/watch?v=W4jaAgjfvG8) made by Matt Muller.

## Streaming ACN (E1.31) Setup
To use the sACN protocol with a ESP8266 and WS2812 led strips, flash your ESP with the
[ESPixelStick](https://github.com/forkineye/ESPixelStick) firmware. 
Building the firmware for Arduino requires numerous libraries listed in the README. 
The process involves flashing the firmware, using gulp to build the HTML for the admin interface,
 and uploading it with the Filesystem Uploader.

The led strip should be connected to the GPIO2 pin (D4 on a NodeMCU). 
Then power up the ESP and connect to the web ui in your browser, 
and got to "Device Setup" and enter the details of your WS2812 strip. 
The color order should be set to "GRB".

From this point the strip can then be configured in the Systematic LED UI's 
Board Manager with settings that match the Device Configuration from ESPixelStick.
