
import React, { useState, useEffect, useRef } from 'react';
import { createRoot } from 'react-dom/client';
import { 
  GoogleGenAI, 
  Modality, 
  GenerateContentResponse,
  Type
} from "@google/genai";

// --- Types & Constants ---
type ToolType = 'think' | 'imagine' | 'speak' | 'animate';

interface ChatMessage {
  role: 'user' | 'model';
  text: string;
}

interface GeneratedMedia {
  type: 'image' | 'video' | 'audio';
  url: string;
  prompt: string;
  timestamp: number;
}

// --- Utils ---
function decode(base64: string) {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number,
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

// --- Icons ---
const Icons = {
  Think: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/><path d="M19 3v4"/><path d="M21 5h-4"/></svg>
  ),
  Imagine: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/></svg>
  ),
  Speak: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M11 5 6 9H2v6h4l5 4V5z"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/></svg>
  ),
  Animate: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect width="20" height="14" x="2" y="3" rx="2"/><path d="M22 7l-3.5 2"/><path d="M22 13l-3.5-2"/><path d="M2 11h16.5"/></svg>
  ),
  Send: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
  ),
  Loader: () => (
    <svg className="animate-spin" xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>
  )
};

// --- App Component ---
const OmniStudio = () => {
  const [activeTool, setActiveTool] = useState<ToolType>('think');
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [mediaGallery, setMediaGallery] = useState<GeneratedMedia[]>([]);
  const [videoStatus, setVideoStatus] = useState<string | null>(null);

  const audioContextRef = useRef<AudioContext | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  const initAudio = () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
    }
  };

  const handleGenerate = async () => {
    if (!input.trim() || loading) return;
    setLoading(true);
    initAudio();

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

      if (activeTool === 'think') {
        const response = await ai.models.generateContent({
          model: 'gemini-3-pro-preview',
          contents: input,
          config: {
            thinkingConfig: { thinkingBudget: 32768 }
          }
        });
        setChatHistory(prev => [...prev, { role: 'user', text: input }, { role: 'model', text: response.text || 'No response.' }]);
        setInput('');
      } 
      
      else if (activeTool === 'imagine') {
        const response = await ai.models.generateContent({
          model: 'gemini-2.5-flash-image',
          contents: { parts: [{ text: input }] },
          config: { imageConfig: { aspectRatio: "16:9" } }
        });

        for (const part of response.candidates[0].content.parts) {
          if (part.inlineData) {
            const url = `data:image/png;base64,${part.inlineData.data}`;
            setMediaGallery(prev => [{ type: 'image', url, prompt: input, timestamp: Date.now() }, ...prev]);
          }
        }
      } 
      
      else if (activeTool === 'speak') {
        const response = await ai.models.generateContent({
          model: 'gemini-2.5-flash-preview-tts',
          contents: [{ parts: [{ text: input }] }],
          config: {
            responseModalities: [Modality.AUDIO],
            speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Zephyr' } } }
          }
        });

        const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
        if (base64Audio && audioContextRef.current) {
          const audioBuffer = await decodeAudioData(decode(base64Audio), audioContextRef.current, 24000, 1);
          const source = audioContextRef.current.createBufferSource();
          source.buffer = audioBuffer;
          source.connect(audioContextRef.current.destination);
          source.start();
          
          // Add to gallery (conceptual for audio)
          setMediaGallery(prev => [{ type: 'audio', url: 'pcm-playback', prompt: input, timestamp: Date.now() }, ...prev]);
        }
      } 
      
      else if (activeTool === 'animate') {
        // Veo requires extra care with API Key
        const hasKey = await (window as any).aistudio.hasSelectedApiKey();
        if (!hasKey) {
          await (window as any).aistudio.openSelectKey();
        }

        const veoAi = new GoogleGenAI({ apiKey: process.env.API_KEY });
        setVideoStatus("Initiating production...");
        
        let operation = await veoAi.models.generateVideos({
          model: 'veo-3.1-fast-generate-preview',
          prompt: input,
          config: { numberOfVideos: 1, resolution: '720p', aspectRatio: '16:9' }
        });

        while (!operation.done) {
          setVideoStatus("Encoding cinematic sequences... (this takes 1-2 mins)");
          await new Promise(resolve => setTimeout(resolve, 8000));
          operation = await veoAi.operations.getVideosOperation({ operation: operation });
        }

        const downloadLink = operation.response?.generatedVideos?.[0]?.video?.uri;
        const videoResponse = await fetch(`${downloadLink}&key=${process.env.API_KEY}`);
        const videoBlob = await videoResponse.blob();
        const videoUrl = URL.createObjectURL(videoBlob);

        setMediaGallery(prev => [{ type: 'video', url: videoUrl, prompt: input, timestamp: Date.now() }, ...prev]);
        setVideoStatus(null);
      }
    } catch (err: any) {
      console.error(err);
      if (err.message?.includes("Requested entity was not found")) {
        alert("API Key session expired. Please re-select your key.");
        await (window as any).aistudio.openSelectKey();
      } else {
        alert("An error occurred: " + err.message);
      }
    } finally {
      setLoading(false);
      setVideoStatus(null);
    }
  };

  return (
    <div className="flex h-full w-full bg-[#0a0a0a] text-gray-200 overflow-hidden">
      {/* Sidebar Navigation */}
      <aside className="w-16 md:w-20 border-r border-white/5 flex flex-col items-center py-8 space-y-8 glass z-20">
        <div className="w-10 h-10 bg-indigo-600 rounded-xl flex items-center justify-center neo-shadow">
          <span className="font-bold text-lg text-white">Ω</span>
        </div>
        
        <nav className="flex flex-col space-y-6">
          <NavButton active={activeTool === 'think'} onClick={() => setActiveTool('think')} icon={<Icons.Think />} label="Think" />
          <NavButton active={activeTool === 'imagine'} onClick={() => setActiveTool('imagine')} icon={<Icons.Imagine />} label="Imagine" />
          <NavButton active={activeTool === 'speak'} onClick={() => setActiveTool('speak')} icon={<Icons.Speak />} label="Speak" />
          <NavButton active={activeTool === 'animate'} onClick={() => setActiveTool('animate')} icon={<Icons.Animate />} label="Animate" />
        </nav>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col relative overflow-hidden">
        {/* Header */}
        <header className="h-16 border-b border-white/5 flex items-center justify-between px-6 glass shrink-0">
          <div className="flex items-center space-x-3">
            <h1 className="text-lg font-semibold tracking-tight text-white capitalize">{activeTool} Workspace</h1>
            {loading && <div className="text-xs text-indigo-400 animate-pulse-subtle flex items-center space-x-2">
              <Icons.Loader />
              <span>Model Processing...</span>
            </div>}
          </div>
          <div className="flex items-center space-x-4">
            <a href="https://ai.google.dev/gemini-api/docs/billing" target="_blank" className="text-xs text-gray-500 hover:text-indigo-400 transition-colors">Billing Docs</a>
          </div>
        </header>

        {/* Workspace Views */}
        <div className="flex-1 flex overflow-hidden">
          {/* Active Workspace Content */}
          <section className="flex-1 flex flex-col p-6 overflow-y-auto relative">
            {activeTool === 'think' ? (
              <div className="flex-1 space-y-6 max-w-4xl mx-auto w-full pb-24">
                {chatHistory.length === 0 && (
                  <div className="h-full flex flex-col items-center justify-center opacity-30 text-center space-y-4">
                    <Icons.Think />
                    <p className="text-sm">Start a deep-thinking session to explore complex ideas.</p>
                  </div>
                )}
                {chatHistory.map((msg, i) => (
                  <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-[85%] p-4 rounded-2xl ${msg.role === 'user' ? 'bg-indigo-600 text-white' : 'bg-white/5 border border-white/10'}`}>
                      <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.text}</p>
                    </div>
                  </div>
                ))}
                <div ref={chatEndRef} />
              </div>
            ) : (
              <div className="flex-1 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 pb-24">
                {mediaGallery.filter(m => 
                  (activeTool === 'imagine' && m.type === 'image') || 
                  (activeTool === 'animate' && m.type === 'video') ||
                  (activeTool === 'speak' && m.type === 'audio')
                ).length === 0 && (
                  <div className="col-span-full h-full flex flex-col items-center justify-center opacity-30 text-center space-y-4">
                    <p className="text-sm">Your generated content will appear here.</p>
                  </div>
                )}
                {mediaGallery
                  .filter(m => (activeTool === 'imagine' && m.type === 'image') || (activeTool === 'animate' && m.type === 'video') || (activeTool === 'speak' && m.type === 'audio'))
                  .map((item, i) => (
                  <div key={i} className="group relative bg-white/5 border border-white/10 rounded-xl overflow-hidden neo-shadow hover:border-white/20 transition-all h-fit">
                    {item.type === 'image' && <img src={item.url} className="w-full aspect-video object-cover" alt={item.prompt} />}
                    {item.type === 'video' && <video src={item.url} controls className="w-full aspect-video object-cover" />}
                    {item.type === 'audio' && <div className="p-6 flex items-center justify-center bg-indigo-900/20"><Icons.Speak /></div>}
                    <div className="p-3">
                      <p className="text-xs text-gray-400 line-clamp-2 italic">"{item.prompt}"</p>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Input Bar */}
            <div className="absolute bottom-6 left-6 right-6 flex items-center justify-center pointer-events-none">
              <div className="max-w-3xl w-full flex flex-col space-y-2 pointer-events-auto">
                {videoStatus && (
                  <div className="bg-black/80 border border-indigo-500/50 rounded-lg p-3 text-xs text-indigo-300 flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Icons.Loader />
                      <span>{videoStatus}</span>
                    </div>
                  </div>
                )}
                <div className="flex items-center bg-[#1a1a1a] border border-white/10 rounded-2xl p-2 shadow-2xl focus-within:border-indigo-500/50 transition-colors">
                  <textarea
                    rows={1}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleGenerate())}
                    placeholder={`Describe your vision for ${activeTool}...`}
                    className="flex-1 bg-transparent border-none outline-none px-4 text-sm resize-none py-2 placeholder:text-gray-600"
                  />
                  <button 
                    onClick={handleGenerate}
                    disabled={loading || !input.trim()}
                    className="w-10 h-10 rounded-xl bg-indigo-600 text-white flex items-center justify-center hover:bg-indigo-500 disabled:opacity-50 disabled:hover:bg-indigo-600 transition-all shrink-0"
                  >
                    {loading ? <Icons.Loader /> : <Icons.Send />}
                  </button>
                </div>
              </div>
            </div>
          </section>

          {/* Right Panel: Context / Details */}
          <aside className="hidden lg:flex w-64 border-l border-white/5 glass flex-col overflow-y-auto">
            <div className="p-4 border-b border-white/5">
              <h3 className="text-xs font-bold text-gray-500 uppercase tracking-widest">Model Settings</h3>
            </div>
            <div className="p-4 space-y-6">
              <SettingItem label="Engine" value={
                activeTool === 'think' ? 'Gemini 3 Pro' : 
                activeTool === 'imagine' ? 'Gemini 2.5 Flash Image' : 
                activeTool === 'speak' ? 'Gemini TTS' : 'Veo 3.1'
              } />
              
              {activeTool === 'think' && <SettingItem label="Think Budget" value="32k Tokens" />}
              {activeTool === 'imagine' && <SettingItem label="Aspect Ratio" value="16:9" />}
              {activeTool === 'speak' && <SettingItem label="Voice" value="Zephyr (Dynamic)" />}
              {activeTool === 'animate' && <SettingItem label="Resolution" value="720p" />}
              
              <div className="pt-4">
                <h4 className="text-[10px] font-bold text-gray-600 uppercase mb-3">Recent Activity</h4>
                <div className="space-y-3">
                  {mediaGallery.slice(0, 5).map((m, i) => (
                    <div key={i} className="flex items-center space-x-3 text-[11px]">
                      <div className="w-1.5 h-1.5 rounded-full bg-indigo-500"></div>
                      <span className="text-gray-400 truncate flex-1">{m.prompt}</span>
                    </div>
                  ))}
                  {mediaGallery.length === 0 && <span className="text-[10px] text-gray-700">No recent activity</span>}
                </div>
              </div>
            </div>
          </aside>
        </div>
      </main>
    </div>
  );
};

const NavButton = ({ active, onClick, icon, label }: { active: boolean, onClick: () => void, icon: React.ReactNode, label: string }) => (
  <button 
    onClick={onClick}
    className={`group relative w-12 h-12 flex items-center justify-center rounded-xl transition-all duration-300 ${active ? 'bg-indigo-600/20 text-indigo-400' : 'text-gray-500 hover:text-gray-300 hover:bg-white/5'}`}
  >
    {icon}
    <span className="absolute left-full ml-4 px-2 py-1 bg-black text-[10px] font-medium rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-50 pointer-events-none border border-white/10">
      {label}
    </span>
    {active && <div className="absolute -left-1 w-1 h-6 bg-indigo-500 rounded-full"></div>}
  </button>
);

const SettingItem = ({ label, value }: { label: string, value: string }) => (
  <div className="space-y-1">
    <p className="text-[10px] text-gray-500 font-medium uppercase">{label}</p>
    <p className="text-xs text-gray-300 font-mono">{value}</p>
  </div>
);

const root = createRoot(document.getElementById('root')!);
root.render(<OmniStudio />);
