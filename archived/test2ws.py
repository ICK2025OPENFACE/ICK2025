import asyncio
import websockets
import datetime


async def send_messages(websocket):
    print("Klient połączony")
    try:
        while True:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            await websocket.send(f"Serwer wysyła aktualny czas: {now}")
            await asyncio.sleep(2)
    except websockets.ConnectionClosed:
        print("Klient rozłączył się")


async def main():
    async with websockets.serve(send_messages, "localhost", 8765):
        print("Serwer WebSocket działa na ws://localhost:8765")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
