import type { Metadata } from 'next'
import { Inter, JetBrains_Mono } from 'next/font/google'
import './globals.css'  // <--- THIS LINE IS CRITICAL

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' })
const jetbrains = JetBrains_Mono({ subsets: ['latin'], variable: '--font-mono' })

export const metadata: Metadata = {
  title: 'Audio DeepFake Detection',
  description: 'Audio Deepfake Detection Platform',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark"> {/* Force dark mode for the futuristic look */}
      <body className={`${inter.variable} ${jetbrains.variable} font-sans bg-slate-950 text-slate-100 antialiased`}>
        {children}
      </body>
    </html>
  )
}