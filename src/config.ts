export const CONFIG = {
    // Canvas & Image Settings
    imageSize: 128, //square image size

    // Diffusion Settings
    timesteps: 1000, // Number of steps T
    betaStart: 0.0001,
    betaEnd: 0.02,

    // Training Settings
    batchSize: 32,
    epochs: 50,
    learningRate: 0.001,

    // UI Settings
    colors: {
        primary: '#646cff',
        background: '#242424',
        text: 'rgba(255, 255, 255, 0.87)'
    }
};
