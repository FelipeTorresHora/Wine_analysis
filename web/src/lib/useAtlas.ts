import { useEffect, useState } from 'react'
import type { Destaque } from '../types/atlas'
import { readDestaques } from './destaques'

export type AtlasStatus = 'loading' | 'ready' | 'error'

export function useAtlas() {
  const [status, setStatus] = useState<AtlasStatus>('loading')
  const [destaques, setDestaques] = useState<Destaque[]>([])

  useEffect(() => {
    let active = true
    const url = `${import.meta.env.BASE_URL}atlas.json`

    fetch(url)
      .then((response) => {
        if (!response.ok) throw new Error(String(response.status))
        return response.json() as Promise<unknown>
      })
      .then((data) => {
        if (!active) return
        const list = readDestaques(data)
        if (!list) {
          setStatus('error')
          return
        }
        setDestaques(list)
        setStatus('ready')
      })
      .catch(() => {
        if (!active) return
        setStatus('error')
      })

    return () => {
      active = false
    }
  }, [])

  return { status, destaques }
}
