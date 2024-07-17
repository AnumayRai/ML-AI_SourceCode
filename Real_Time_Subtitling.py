import React, { useState } from 'react';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';

const RealTimeTranscription = () => {
  const [transcript, setTranscript] = useState('');
  const { transcript: liveTranscript } = useSpeechRecognition();

  if (!SpeechRecognition.browserSupportsSpeechRecognition()) {
    return <p>Browser does not support speech recognition.</p>;
  }

  React.useEffect(() => {
    setTranscript(liveTranscript);
  }, [liveTranscript]);

  return (
    <div>
      <button onClick={SpeechRecognition.startListening}>Start</button>
      <button onClick={SpeechRecognition.stopListening}>Stop</button>
      <p>{transcript}</p>
    </div>
  );
};

export default RealTimeTranscription;

import React from 'react';

const SubtitleDisplay = ({ subtitle }) => {
  return (
    <div style={{ position: 'absolute', bottom: '10px', backgroundColor: 'rgba(0, 0, 0, 0.5)', color: 'white', padding: '10px' }}>
      {subtitle}
    </div>
  );
};

export default SubtitleDisplay;

import React from 'react';
import RealTimeTranscription from './RealTimeTranscription';
import SubtitleDisplay from './SubtitleDisplay';

const App = () => {
  const [transcript, setTranscript] = React.useState('');

  return (
    <div>
      <RealTimeTranscription setTranscript={setTranscript} />
      <SubtitleDisplay subtitle={transcript} />
    </div>
  );
};

export default App;
