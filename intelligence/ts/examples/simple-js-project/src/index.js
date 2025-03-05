import { FlowerIntelligence } from '@flwr/flwr';

const fi = FlowerIntelligence.instance;
fi.remoteHandoff = true;
fi.apiKey = process.env.FI_API_KEY ?? 'REPLACE_HERE';

async function main() {
  const response = await fi.chat({
    messages: [
      { role: 'system', content: 'You are a helpful assistant' },
      { role: 'user', content: 'How are you?' },
    ],
    forceRemote: true,
  });
  if (!response.ok) {
    console.error(`${response.failure.code}: ${response.failure.description}`);
    process.exit(1);
  } else {
    console.log(response.content);
  }
}

await main().then().catch();
