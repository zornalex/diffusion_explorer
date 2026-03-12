export const CONFIG = {
    // Canvas & Image Settings
    imageSize: 32, //square image size

    // Diffusion Settings
    timesteps: 400, // Number of steps T
    betaStart: 0.0001,
    betaEnd: 0.02,

    // Training Settings
    batchSize: 8,            // small batches → less GPU↔CPU overhead per step
    trainingSteps: 2000,     // 4× more steps → ~8 visits per (t,image) pair, much better convergence
    learningRate: 0.001,     // Adam lr for fine-tuning
    timeDim: 16,             // sinusoidal time-embedding dimension (8 sin + 8 cos frequencies)

    // UI Settings
    colors: {
        primary: '#646cff',
        background: '#242424',
        text: 'rgba(255, 255, 255, 0.87)'
    }
};
