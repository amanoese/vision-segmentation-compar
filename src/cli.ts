import { parseArgs } from 'node:util';
import {
  ALL_METHODS,
  methodsForConfig,
  detect,
  loadConfig,
  prepare,
  render,
  renderVideos,
} from './video.js';
import type { Method } from './types.js';

const command = process.argv[2];
const { values } = parseArgs({
  args: process.argv.slice(3),
  options: {
    config: { type: 'string', default: 'configs/skin-sns-openrouter-agent.json' },
    method: { type: 'string' },
  },
});
try {
  const config = await loadConfig(values.config!);
  if (command === 'prepare') console.log(JSON.stringify(await prepare(config), null, 2));
  else if (command === 'detect') {
    if (!ALL_METHODS.includes(values.method as Method))
      throw new Error('Specify --method sam|sam-agent-fixed|sam-agent|jev');
    await detect(config, values.method as Method);
  } else if (command === 'render') console.log(JSON.stringify(await render(config), null, 2));
  else if (command === 'render-videos')
    console.log(JSON.stringify(await renderVideos(config), null, 2));
  else if (command === 'run') {
    for (const method of methodsForConfig(config)) await detect(config, method);
    console.log(JSON.stringify(await render(config), null, 2));
  } else
    throw new Error(
      'Commands: prepare | detect --method sam|sam-agent-fixed|sam-agent|jev | render | render-videos | run [--config FILE]',
    );
} catch (error) {
  console.error(error instanceof Error ? error.message : 'Comparison failed');
  process.exitCode = 1;
}
