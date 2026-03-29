const INDEXTTS2_URL     = process.env.INDEXTTS2_URL     ?? 'http://localhost:8031';
const INDEXTTS1_5_URL   = process.env.INDEXTTS1_5_URL   ?? 'http://localhost:8032';
const FISHAUDIO_URL     = process.env.FISHAUDIO_URL     ?? 'http://localhost:8033';
const MFA_CONFORMER_URL = process.env.MFA_CONFORMER_URL ?? 'http://localhost:8011';
const AASIST_URL        = process.env.AASIST_URL        ?? 'http://localhost:8012';
const SPEECHPROMPT_URL  = process.env.SPEECHPROMPT_URL  ?? 'http://localhost:8013';
const ASR_URL           = process.env.ASR_URL           ?? 'http://localhost:8021';

export const BACKEND_URLS = {
  TTS: {
    IndexTTS2: {
      main:   INDEXTTS2_URL,
      status: `${INDEXTTS2_URL}/status`,
      tts:    `${INDEXTTS2_URL}/tts`,
    },
    'IndexTTS1.5': {
      main:   INDEXTTS1_5_URL,
      status: `${INDEXTTS1_5_URL}/status`,
      tts:    `${INDEXTTS1_5_URL}/tts`,
    },
    FishAudio: {
      main:   FISHAUDIO_URL,
      status: `${FISHAUDIO_URL}/status`,
      tts:    `${FISHAUDIO_URL}/tts`,
    },
    // Add more TTS models here
  },
  DETECT: {
    'MFA-Conformer': {
      main:   MFA_CONFORMER_URL,
      status: `${MFA_CONFORMER_URL}/status`,
      detect: `${MFA_CONFORMER_URL}/detect`,
    },
    AASIST: {
      main:   AASIST_URL,
      status: `${AASIST_URL}/status`,
      detect: `${AASIST_URL}/detect`,
    },
    SpeechPrompt: {
      main:   SPEECHPROMPT_URL,
      status: `${SPEECHPROMPT_URL}/status`,
      detect: `${SPEECHPROMPT_URL}/detect`,
    },
  },
  ASR: {
    main:       ASR_URL,
    status:     `${ASR_URL}/status`,
    transcribe: `${ASR_URL}/asr`,
  },
};

export default BACKEND_URLS;
