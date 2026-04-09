// EPL Oracle v4.1 — Logger (pino)

import pino from 'pino';

const isDev = process.env.NODE_ENV !== 'production' && process.env.LOG_FORMAT !== 'json';

export const logger = pino({
  level: process.env.LOG_LEVEL ?? 'info',
  ...(isDev
    ? {
        transport: {
          target: 'pino-pretty',
          options: { colorize: true, translateTime: 'HH:MM:ss', ignore: 'pid,hostname' },
        },
      }
    : {}),
});
