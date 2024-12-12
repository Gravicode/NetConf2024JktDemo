using OpenAI.RealtimeConversation;
using OpenAI;
using System.ClientModel;
using System.Text;
using System.Diagnostics;
using Microsoft.Extensions.AI;
using System.ComponentModel;
#pragma warning disable OPENAI002 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.

namespace TalkingBot.Helpers
{

    public class MathParamInput
    {
        public string math_question { get; set; }
    }

    public class LogMessage : EventArgs
    {
        public DateTime Created { get; set; } = DateTime.Now;
        public string Message { get; set; }

        public bool NewLine { get; set; } = true;
    }
    public class RealtimeVoiceBot
    {
        MathPlugin mathPlugin { set; get; }
        Thread conversationThread { get; set; }
        public EventHandler<LogMessage> LogMessageReceived;
        CancellationTokenSource cancellationTokenSource;
        public bool IsRunning { get; set; } = false;
        public RealtimeVoiceBot()
        {
            mathPlugin = new();
        }
        public async Task Stop()
        {
            if (!IsRunning)
            {
                WriteLog("Bot is not running.. cannot stop");
                return;
            }
            WriteLog("Trying to stop bot..");
            cancellationTokenSource.Cancel();
        }

        void WriteLog(string message = "", bool Newline = true)
        {
            var additionalEnd = string.Empty;
            var Msg = message;
            if (Newline)
            {
                additionalEnd = "\n";
                Msg = string.IsNullOrEmpty(message) ? $"---------------{additionalEnd}" : $"{DateTime.Now.ToString("dd-MMM-yy HH:mm:ss")} => {message}{additionalEnd}";
            }

            Debug.WriteLine(Msg);
            LogMessageReceived?.Invoke(this, new() { Message = Msg, NewLine = Newline });
        }

        private RealtimeConversationClient GetConfiguredClient()
        {
            return GetConfiguredClientForOpenAIWithKey(AppConstants.OpenAIKey);
        }

        private RealtimeConversationClient GetConfiguredClientForOpenAIWithKey(string oaiApiKey)
        {
            string oaiEndpoint = "https://api.openai.com/v1";
            WriteLog($" * Connecting to OpenAI endpoint (OPENAI_ENDPOINT): {oaiEndpoint}");
            WriteLog($" * Using API key (OPENAI_API_KEY): {oaiApiKey[..5]}**");

            OpenAIClient aoaiClient = new(new ApiKeyCredential(oaiApiKey));
            return aoaiClient.GetRealtimeConversationClient(AppConstants.ModelId);
        }
        private async Task HandleToolCallsAsync(RealtimeConversationSession session, ConversationUpdate update, AIFunction[] tools)
        {
            switch (update)
            {
                case ConversationItemStreamingFinishedUpdate itemFinished:
                    // If we need to call a tool to update the model, do so
                    if (!string.IsNullOrEmpty(itemFinished.FunctionName) && await itemFinished.GetFunctionCallOutputAsync(tools) is { } output)
                    {
                        await session!.AddItemAsync(output);
                    }
                    break;

                case ConversationResponseFinishedUpdate responseFinished:
                    // If we added one or more function call results, instruct the model to respond to them
                    if (responseFinished.CreatedItems.Any(item => !string.IsNullOrEmpty(item.FunctionName)))
                    {
                        await session!.StartResponseAsync();
                    }
                    break;
            }
        }
        void CallFuntionLog(string FunctionName, string Parameter="")
        {
            WriteLog($"function call: {FunctionName} => [{Parameter}]");
        }
        public async Task Start()
        {
            if (IsRunning)
            {
                WriteLog("Bot is already running..");
                return;
            }
            cancellationTokenSource = new();
            var token = cancellationTokenSource.Token;
            conversationThread = new Thread(async () =>
            {
                // First, we create a client according to configured environment variables (see end of file) and then start
                // a new conversation session.
                RealtimeConversationClient client = GetConfiguredClient();
                using RealtimeConversationSession session = await client.StartConversationSessionAsync();

                // We'll add a simple function tool that enables the model to interpret user input to figure out when it
                // might be a good time to stop the interaction.
                ConversationFunctionTool finishConversationTool = new()
                {
                    Name = "user_wants_to_finish_conversation",
                    Description = "Invoked when the user says goodbye, expresses being finished, or otherwise seems to want to stop the interaction.",
                    Parameters = BinaryData.FromString("{}"),
                };

                ConversationFunctionTool GetCurrentUtcTimeTool = new()
                {
                    Name = "get_current_utc_time",
                    Description = "Retrieves the current time in UTC.",
                    Parameters = BinaryData.FromString("{}"),
                };

                ConversationFunctionTool MathTool = new()
                {
                    Name = "calculate_math",
                    Description = "Translate a math problem into a expression that can be executed using .net NCalc library",
                    Parameters = BinaryData.FromString(
                        """
                        {
                          "type": "object",
                          "properties": {
                            "math_question": {
                              "type": "string",
                              "description": "Question with math problem"
                            }
                          },
                          "required": ["math_question"]
                        }
                        """),
                };

                var getUTC = AIFunctionFactory.Create(() => {
                    CallFuntionLog(GetCurrentUtcTimeTool.Name);
                    return DateTime.UtcNow.ToString("R"); 
                }, GetCurrentUtcTimeTool.Name, GetCurrentUtcTimeTool.Description);

                var calculateMath = AIFunctionFactory.Create(async ([Description("Question with math problem")] string math_question) => {
                    CallFuntionLog(MathTool.Name, math_question);
                    return await mathPlugin.Calculate(math_question); 
                }, MathTool.Name, MathTool.Description);

                var finishConversation = AIFunctionFactory.Create(() => { 
                    CallFuntionLog(finishConversationTool.Name);
                    WriteLog($" <<< Finish tool invoked -- ending conversation!");
                    if (cancellationTokenSource != null) cancellationTokenSource.Cancel();
                }, finishConversationTool.Name, finishConversationTool.Description);

                AIFunction[] toolsAI = [finishConversation, getUTC, calculateMath];
                // Now we configure the session using the tool we created along with transcription options that enable input
                // audio transcription with whisper.
                var opt = new ConversationSessionOptions()
                {
                    Instructions = $"""
                Kamu adalah virtual asisten bernama Siti Kodingwati, kamu tinggal di jakarta, kamu ramah dan lucu, kamu sedang menjadi host di acara .NET Conf 2024, speaker yang akan mengisi acara antara lain:                                
                1. "Aspire in .NET 9.0" oleh Eriawan 
                2. "AI dan .NET is Great" oleh Fadhil
                3. "Performance enhancement in .NET 9.0" oleh Ridi
                4. "MAUI in .NET 9.0" oleh Eric Kurniawan
                """,
                    Voice = ConversationVoice.Alloy,
                    //Tools = tools,
                    InputTranscriptionOptions = new()
                    {
                        Model = "whisper-1",
                    },
                };
                foreach (var itemTool in toolsAI)
                {
                    opt.Tools.Add(itemTool.ToConversationFunctionTool());
                }

                await session.ConfigureSessionAsync(opt);

                // For convenience, we'll proactively start playback to the speakers now. Nothing will play until it's enqueued.
                SpeakerOutput speakerOutput = new();
                var outputStringBuilder = new StringBuilder();
                // With the session configured, we start processing commands received from the service.
                await foreach (ConversationUpdate update in session.ReceiveUpdatesAsync())
                {
                    // session.created is the very first command on a session and lets us know that connection was successful.
                    if (update is ConversationSessionStartedUpdate)
                    {
                        WriteLog($" <<< Connected: session started");
                        // This is a good time to start capturing microphone input and sending audio to the service. The
                        // input stream will be chunked and sent asynchronously, so we don't need to await anything in the
                        // processing loop.
                        _ = Task.Run(async () =>
                        {
                            using MicrophoneAudioStream microphoneInput = MicrophoneAudioStream.Start();
                            IsRunning = true;
                            WriteLog($" >>> Listening to microphone input");
                            WriteLog($" >>> (Just tell the app you're done to finish)");
                            WriteLog();
                            await session.SendInputAudioAsync(microphoneInput);

                        });
                    }

                    if (token.IsCancellationRequested)
                    {
                        WriteLog($" <<< Request to stop!");
                        break;
                    }

                    // input_audio_buffer.speech_started tells us that the beginning of speech was detected in the input audio
                    // we're sending from the microphone.
                    if (update is ConversationInputSpeechStartedUpdate)
                    {
                        WriteLog($" <<< Start of speech detected");
                        // Like any good listener, we can use the cue that the user started speaking as a hint that the app
                        // should stop talking. Note that we could also track the playback position and truncate the response
                        // item so that the model doesn't "remember things it didn't say" -- that's not demonstrated here.
                        speakerOutput.ClearPlayback();
                    }

                    // input_audio_buffer.speech_stopped tells us that the end of speech was detected in the input audio sent
                    // from the microphone. It'll automatically tell the model to start generating a response to reply back.
                    if (update is ConversationInputSpeechFinishedUpdate)
                    {
                        WriteLog($" <<< End of speech detected");
                    }

                    // conversation.item.input_audio_transcription.completed will only arrive if input transcription was
                    // configured for the session. It provides a written representation of what the user said, which can
                    // provide good feedback about what the model will use to respond.
                    if (update is ConversationInputTranscriptionFinishedUpdate transcriptionFinishedUpdate)
                    {
                        WriteLog($" >>> USER: {transcriptionFinishedUpdate.Transcript}");

                    }

                    // response.audio.delta provides incremental output audio generated by the model talking. Here, we
                    // immediately enqueue it for playback on the active speaker output.
                    if (update is ConversationItemStreamingPartDeltaUpdate audioDeltaUpdate)//ConversationAudioDeltaUpdate
                    {
                        if(audioDeltaUpdate.AudioBytes!=null)
                            speakerOutput.EnqueueForPlayback(audioDeltaUpdate.AudioBytes);
                        outputStringBuilder.Append(audioDeltaUpdate.Text ?? audioDeltaUpdate.AudioTranscript);
                        //WriteLog(audioDeltaUpdate.AudioTranscript);
                    }
                    
                    if (update is ConversationResponseFinishedUpdate responseFinished)
                    {
                        // Happens when a "response turn" is finished
                        WriteLog(outputStringBuilder.ToString());
                        outputStringBuilder.Clear();
                    }
                    // error commands, as the name implies, are raised when something goes wrong.
                    if (update is ConversationErrorUpdate errorUpdate)
                    {

                        WriteLog($" <<< ERROR: {errorUpdate.Message}");
                        WriteLog(errorUpdate.GetRawContent().ToString());
                        break;
                    }
                    await HandleToolCallsAsync(session, update, toolsAI);
                }
                IsRunning = false;
                WriteLog("Conversation is finished.");
            });
            conversationThread.Start();

        }

    }
}
#pragma warning restore OPENAI002 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
