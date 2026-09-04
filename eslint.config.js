import js from '@eslint/js';
import globals from 'globals';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  { ignores: ['dist', 'node_modules', 'public/ort'] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  { files: ['scripts/**/*.mjs', 'tests/**/*.mjs'], languageOptions: { globals: globals.node } },
  {
    files: ['src/**/*.{ts,tsx}'],
    languageOptions: {
      globals: globals.browser,
    },
  },
);
