const express = require('express');
const http = require('http');
const io = require('socket.io');
const app = express();
const server = http.createServer(app);
const ioServer = io(server);

app.use(express.static('client/build'));

ioServer.on('connection', (socket) => {
  socket.on('join-room', (roomId, userId) => {
    socket.join(roomId);
    socket.to(roomId).broadcast.emit('user-connected', userId);

    socket.on('disconnect', () => {
      socket.to(roomId).broadcast.emit('user-disconnected', userId);
    });
  });

  socket.on('offer', (roomId, offer) => {
    socket.to(roomId).emit('offer', offer);
  });

  socket.on('answer', (roomId, answer) => {
    socket.to(roomId).emit('answer', answer);
  });

  socket.on('ice-candidate', (roomId, candidate) => {
    socket.to(roomId).emit('ice-candidate', candidate);
  });
});

const PORT = process.env.PORT || 5000;
server.listen(PORT, () => console.log(`Server running on port ${PORT}`));
import React, { useEffect, useRef, useState } from 'react';
import io from 'socket.io-client';
import Peer from 'simple-peer';
import { recognizeStream } from './speechToText';
import TextField from '@material-ui/core/TextField';
import Button from '@material-ui/core/Button';

const socket = io.connect('/');

const VideoCall = () => {
  const [roomId, setRoomId] = useState('');
  const [userId, setUserId] = useState('');
  const [remoteStream, setRemoteStream] = useState(null);
  const userVideo = useRef();
  const partnerVideo = useRef();
  const [captions, setCaptions] = useState([]);

  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: true, audio: true }).then((stream) => {
      if (userVideo.current) {
        userVideo.current.srcObject = stream;
      }

      socket.emit('join-room', roomId, userId);

      socket.on('user-connected', (userId) => {
        const peer = new Peer({
          initiator: true,
          trickle: false,
          stream,
        });

        peer.on('signal', (data) => {
          socket.emit('offer', roomId, data);
        });

        peer.on('stream', (currentStream) => {
          setRemoteStream(currentStream);
          if (partnerVideo.current) {
            partnerVideo.current.srcObject = currentStream;
          }

          recognizeStream(currentStream.getAudioTracks()[0])
            .then((transcriptions) => {
              setCaptions((prevCaptions) => [...prevCaptions, transcriptions]);
            })
            .catch((error) => {
              console.error(error);
            });
        });
      });

      // ... (handle other WebRTC signaling events)
    });
  }, [roomId, userId]);

  return (
    <div>
      {/* Inputs for roomId and userId */}
      {/* Video elements for local and remote streams */}
      {/* Display captions */}
    </div>
  );
};

export default VideoCall;
