import asyncio
import websockets


async def echo(websocket):
    print("Klient połączony")
    try:
        async for message in websocket:
            print(f"Otrzymano: {message}")
            # Odesłanie tej samej wiadomości z powrotem do klienta
            await websocket.send(f"Echo: {message}")
    except websockets.ConnectionClosed:
        print("Klient rozłączył się")


# Uruchom serwer na porcie 8765
async def main():
    async with websockets.serve(echo, "localhost", 8765):
        print("Serwer WebSocket działa na ws://localhost:8765")
        await asyncio.Future()  # Poczekaj na zatrzymanie serwera


if __name__ == "__main__":
    asyncio.run(main())
