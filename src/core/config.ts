/** Semantic version — bump on every feature branch merge to main. */
export const APP_VERSION = '0.2.1';

export const CONFIG = {
    imageSize: 32,
    timesteps: 400,
    betaStart: 0.0001,
    betaEnd: 0.02,
    batchSize: 8,
    trainingSteps: 500,       // was 2000; pretrained model converges in 300-500
    learningRate: 0.001,
    timeDim: 16,

    // Training Quality Settings
    lrWarmupSteps: 10,
    emaDecay: 0.999,
    earlyStopPatience: 50,
    earlyStopThreshold: 0.01,

    // Training Loop Performance
    nextFrameInterval: 20,
    debugFrameInterval: 50,

    colors: {
        primary: '#646cff',
        background: '#242424',
        text: 'rgba(255, 255, 255, 0.87)'
    }
};
