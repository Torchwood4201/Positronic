# SATPN

A modern TypeScript/React-based AI cognitive interface project.

## Project Structure

- **index.html**: Main HTML entry point
- **src/**: Source code
  - **App.tsx**: Main React app
  - **components/**: UI components (AIFace, CognitiveInterface, etc.)
  - **core/**: Core logic (CognitiveOS, PositronicCore, UnifiedModelHub, etc.)
  - **hooks/**: Custom React hooks
  - **types/**: TypeScript type definitions
- **public/**: (if present) Static assets
- **package.json**: Project metadata and dependencies
- **.gitignore**: Files/folders ignored by git

## Getting Started

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/satpn.git
   cd satpn
   ```
2. **Install dependencies:**
   ```sh
   npm install
   ```
3. **Run the app locally:**
   ```sh
   npm run dev
   ```
   Open `index.html` directly in your browser if you prefer (no server required).

## Building for Production

```sh
npm run build
```
The output will be in the `dist/` folder (add to `.gitignore` by default).

## GitHub Pages Deployment

1. Build the project: `npm run build`
2. Push the contents of the `dist/` folder to your `gh-pages` branch or use a deploy action.
3. Ensure all asset paths are relative.

## Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions or support, open an issue on GitHub. 